with 
    base as (
        select s.id as `ID`,
               s.erp_cono as `Cono`,
               upper(s.erp_product_number) as `Vallen ID`,
               upper(replace(i.descrip, ';', ' ')) as `FullDesc`,
               -- s.category_l4_id,
               upper(s.global_vendor_number) as `Global Vendno`,
               upper(i.prodcat) as `ICSP Prodcat`,
               upper(i.zzunspsc) as `UNSPSC`,
               concat(s.erp_cono, '_',  upper(s.global_vendor_number), '_', upper(w.prodline)) as `Prodline`
        from prd_bronze.stibo_ftp.sku s 
        inner join prd_bronze.openedge.icsp i 
        on s.erp_product_number = i.prod 
        left join prd_bronze.openedge.icsw w 
        on s.erp_product_number = w.prod 
           and w.st_row_current=1 
           and w.cono=1 
           and upper(w.whse) in ('ZDAT', 'FC')
        where s.st_row_current=1 
            -- and s.category_l4_id is not null 
            and s.category_l4_id is null 
            and i.st_row_current=1 
            and i.cono in (1,55) 
            and i.descrip is not null
    )

select *
from base b