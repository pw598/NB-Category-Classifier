with
    base as (
        select s.id as `ID`,
               s.erp_cono as `Cono`,
               upper(s.erp_product_number) as `Vallen ID`,
               upper(replace(i.descrip, ';', ' ')) as `FullDesc`,
               s.category_l4_id,
               upper(s.global_vendor_number) as `Global Vendno`,
               upper(i.prodcat) as `ICSP Prodcat`,
               upper(i.zzunspsc) as `UNSPSC`,
               concat(s.erp_cono, '_',  upper(s.global_vendor_number), '_', upper(w.prodline)) as `Prodline`
        from prd_bronze.stibo_ftp.sku s
        inner join prd_bronze.openedge.icsp i
        on s.erp_product_number = i.prod
        left join prd_bronze.openedge.icsw w
        on s.erp_product_number = w.prod
           and w.st_row_current = 1
           and w.cono = 1
           and upper(w.whse) in ('ZDAT', 'FC')
        where s.st_row_current = 1
            and s.category_l4_id is not null
            and i.st_row_current = 1
            and i.cono in (1, 55)
            and i.descrip is not null
    ),

    hierarchy as (
        select
            l1.id   as `Level 1 ID`,
            l1.name as `Level 1 Name`,
            l2.id   as `Level 2 ID`,
            l2.name as `Level 2 Name`,
            l3.id   as `Level 3 ID`,
            l3.name as `Level 3 Name`,
            l4.id   as `Level 4 ID`,
            l4.name as `Level 4 Name`
        from prd_bronze.stibo_ftp.hierarchy l1
        left join prd_bronze.stibo_ftp.hierarchy l2
            on l2.parent_id = l1.id and l2.level = 2 and l2.st_row_current = 1
        left join prd_bronze.stibo_ftp.hierarchy l3
            on l3.parent_id = l2.id and l3.level = 3 and l3.st_row_current = 1
        left join prd_bronze.stibo_ftp.hierarchy l4
            on l4.parent_id = l3.id and l4.level = 4 and l4.st_row_current = 1
        where l1.level = 1
            and l1.st_row_current = 1
    )

select *
from base b
inner join hierarchy h
on b.category_l4_id = h.`Level 4 ID`
where h.`Level 4 ID` is not null