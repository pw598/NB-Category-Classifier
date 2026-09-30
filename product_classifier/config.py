"""Configuration objects for the product-category classifier.

Everything the user is expected to tune lives here. Nothing else in the
package should contain a hard-coded column name, threshold, or
hyper-parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from .paths import data_path, output_dir as default_output_dir

# Floor applied to probabilities before taking logs, to avoid log(0).
EPS = 1e-12

PredictionMode = Literal["joint_path", "per_level"]


# ---------------------------------------------------------------------------
# Data / schema
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Where the data is and which columns mean what.

    level_columns is the *full* hierarchy, deepest last. How many of those
    levels a run actually uses is controlled by ModelConfig.n_levels, so you
    can point at one golden set and still train an L1-only model.
    """

    # Repo-relative, on a cluster and locally alike: <repo>/data.
    # Override with the PRODUCT_CLASSIFIER_DATA_DIR env var, or just set
    # an absolute path here. See product_classifier.paths.
    path: str = field(default_factory=lambda: data_path("cleaned_data.txt"))
    file_format: Literal["csv", "excel", "delta", "spark_table"] = "csv"
    delimiter: str = "|"
    text_col: str = "FullDesc"
    id_col: Optional[str] = "Vallen ID"
    level_columns: List[str] = field(
        default_factory=lambda: [
            "Level 1 Name",
            "Level 2 Name",
            "Level 3 Name",
            "Level 4 Name",
        ]
    )
    # Rows with a null description can never be classified; drop them.
    drop_null_text: bool = True
    # Rows with a null label at any *used* level are unusable for training.
    drop_null_labels: bool = True


@dataclass
class SplitConfig:
    """Train/test split behaviour.

    min_path_count is the rare-path threshold: hierarchy paths seen fewer
    than this many times are removed before splitting. Stratification
    requires at least 2 members per class, so anything below 2 will break a
    stratified split.
    """

    test_size: float = 0.25
    random_state: int = 42
    stratify: bool = True
    min_path_count: int = 20


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@dataclass
class VectorizerConfig:
    """Parameters shared by CountVectorizer and TfidfVectorizer.

    kind selects which vectorizer is built; parameters that don't apply to
    the chosen kind are ignored.
    """

    kind: Literal["tfidf", "count"] = "tfidf"
    max_features: Optional[int] = 40000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: Union[int, float] = 2
    max_df: Union[int, float] = 0.95
    stop_words: Optional[str] = "english"
    lowercase: bool = True
    binary: bool = False  # CountVectorizer only
    sublinear_tf: bool = True  # TfidfVectorizer only
    norm: Optional[str] = "l2"  # TfidfVectorizer only
    use_idf: bool = True  # TfidfVectorizer only
    # Treat a bigram's two terms as a set: "ball valve" and "valve ball"
    # become the same feature. Descriptions are keyword lists rather than
    # prose, and the cleaning pass changes which words end up adjacent, so
    # word order carries little signal. Only two-term n-grams are folded;
    # unigrams and trigrams keep their order. Needs 2 in ngram_range to do
    # anything.
    unordered_bigrams: bool = False
    # Anything else you want to pass straight through to sklearn.
    extra: Dict[str, Any] = field(default_factory=dict)

    def _check_bigrams(self):
        """Called from build() as well as here: grid search assigns fields
        after construction, so __post_init__ alone would miss a candidate
        that pairs unordered_bigrams with an ngram_range having no bigrams.
        """
        lo, hi = tuple(self.ngram_range)
        if self.unordered_bigrams and not (lo <= 2 <= hi):
            raise ValueError(
                f"unordered_bigrams=True but ngram_range={self.ngram_range} "
                "produces no bigrams, so the setting would do nothing. Use "
                "ngram_range=(1, 2) or (2, 2), or leave unordered_bigrams False."
            )

    def __post_init__(self):
        self._check_bigrams()

    def build(self):
        self._check_bigrams()
        common = dict(
            max_features=self.max_features,
            ngram_range=tuple(self.ngram_range),
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=self.stop_words,
            lowercase=self.lowercase,
            **self.extra,
        )
        if self.kind == "count":
            params = dict(binary=self.binary, **common)
        elif self.kind == "tfidf":
            params = dict(
                sublinear_tf=self.sublinear_tf,
                norm=self.norm,
                use_idf=self.use_idf,
                **common,
            )
        else:
            raise ValueError(
                f"Unknown vectorizer kind: {self.kind!r}. Use 'count' or 'tfidf'."
            )

        if self.unordered_bigrams:
            from .vectorizers import build_unordered

            return build_unordered(self.kind, **params)
        if self.kind == "count":
            return CountVectorizer(**params)
        return TfidfVectorizer(**params)


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

@dataclass
class NBConfig:
    """MultinomialNB parameters, applied identically to every level's model.

    alpha: additive (Laplace/Lidstone) smoothing.
    fit_prior: learn class priors from data; False gives uniform priors.
    class_prior: explicit priors, if you have them.
    """

    alpha: float = 0.01
    fit_prior: bool = True
    class_prior: Optional[List[float]] = None

    def build(self) -> MultinomialNB:
        return MultinomialNB(
            alpha=self.alpha,
            fit_prior=self.fit_prior,
            class_prior=self.class_prior,
        )


@dataclass
class ModelConfig:
    """How the per-level models are combined into a prediction.

    n_levels: how deep to go. 1 = L1 only, 2 = L1+L2, ... 4 = L1-L4. The
        classifier slices DataConfig.level_columns to this depth, so an
        L1-only run and an L1-L4 run read the same golden set.

    prediction_mode:
        "joint_path"  - score every hierarchy path observed in training by
            summing the per-level log-probabilities, then take the best
            path. Predictions are guaranteed to be a real path (an L4 can
            never be paired with an L1 it never appears under), and the
            reported Probability is the joint probability of the whole path.
        "per_level"   - take each level's own argmax independently. Faster,
            but the levels are only coincidentally consistent with each
            other and there is no single path probability.

        Terminology note: in the hierarchical-classification literature the
        first is a "global"/path-constrained prediction and the second is
        "flat" per-level prediction. Either pair of words reads fine in a
        report; "joint path vs. per-level" is the clearest for a non-ML
        audience.

    batch_size: rows scored per inference batch. Bounds peak memory,
        independent of test-set size.

    fit_chunk_size: rows per partial_fit chunk during training. sklearn's
        MultinomialNB.fit densifies the labels into an (n_rows x n_classes)
        float64 array, which is the largest allocation in the whole
        pipeline once a level has many classes -- at 448k rows and 2,500
        L4 classes that single array is ~9 GB. Fitting in chunks bounds it
        to (fit_chunk_size x n_classes) instead. Naive Bayes fitting is
        pure count accumulation, so chunked fitting is mathematically
        identical to a single fit() call: same model, same accuracy, every
        row still processed exactly once. Set to None to use a single
        fit() call.

    normalise_probabilities: what the reported Probability means, and the
        single switch that puts predict() and predict_prefix_probabilities()
        on a common scale.

        None (the default)
            each method keeps its own historical convention: predict()
            reports the raw product of the per-level posteriors, while
            predict_prefix_probabilities() normalises. Convenient, but the
            two are then not comparable, so a threshold read off one does
            not mean the same thing on the other.

        True
            both normalise: the score is divided by the total across every
            valid path, so candidates sum to 1. Comparable across depths,
            and at full depth the two methods report exactly the same
            number. predict() additionally returns a 'Path Evidence'
            column carrying the unnormalised figure, since normalising
            alone would hide a product that fits nothing well.

        False
            both report raw products. Confidence then shrinks mechanically
            with depth, but the calibration buckets and the cascade
            thresholds are at least on one scale.

        Only joint_path is affected. per_level has no candidate path set to
        normalise over.

    dtype: float32 halves the memory of the (batch x n_paths) score matrix
        versus float64, at negligible accuracy cost.
    """

    n_levels: int = 4
    prediction_mode: PredictionMode = "joint_path"
    normalise_probabilities: Optional[bool] = None
    batch_size: int = 2000
    fit_chunk_size: Optional[int] = 20000
    dtype: str = "float32"
    vectorizer: VectorizerConfig = field(default_factory=VectorizerConfig)
    nb: NBConfig = field(default_factory=NBConfig)


# ---------------------------------------------------------------------------
# Evaluation / reporting
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Evaluation and reporting options."""

    # Width of the confidence buckets used for the calibration report.
    probability_bucket_width: float = 0.1
    # Also score the "per_level" baseline alongside a joint_path run, so the
    # two modes can be compared in one pass. Ignored in per_level runs.
    compare_against_per_level: bool = True
    # Optional: rows below this joint probability are flagged for review.
    review_threshold: Optional[float] = None
    output_dir: str = field(default_factory=lambda: str(default_output_dir()))
    # "csv" needs nothing extra; "parquet" needs pyarrow or fastparquet and
    # is worth it once the comparison frame gets large. If parquet is
    # requested but unavailable, save_outputs falls back to csv with a
    # warning rather than failing at the end of a long run.
    output_format: Literal["csv", "parquet", "excel"] = "csv"


@dataclass
class PipelineConfig:
    """Top-level container. One of these fully describes a run."""

    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    # Cleaning is optional: leave as None to use an already-cleaned file.
    clean_text: bool = False

    def active_levels(self) -> List[str]:
        """The level columns this run will actually use."""
        n = self.model.n_levels
        levels = self.data.level_columns
        if not 1 <= n <= len(levels):
            raise ValueError(
                f"n_levels={n} but only {len(levels)} level columns are configured."
            )
        return levels[:n]

    def to_dict(self) -> Dict[str, Any]:
        """Flat-ish dict, handy for MLflow params or a run log."""
        return asdict(self)
