    select 
    base.oid_usuario -- Identificar único do usuário
    ,base.oid_conta -- Identificador único da conta
    ,base.num_conta
    ,case when base.NUM_ANO_MES = 202201 and (QTD_TRANSACAO) > 0 then 1 else 0 end as FLG_202201
    ,case when base.NUM_ANO_MES = 202202 and (QTD_TRANSACAO) > 0 then 1 else 0 end as FLG_202202
    ,case when base.NUM_ANO_MES = 202203 and (QTD_TRANSACAO) > 0 then 1 else 0 end as FLG_202203
    ,case when base.NUM_ANO_MES = 202204 and (QTD_TRANSACAO) > 0 then 1 else 0 end as FLG_202204
    ,case when base.NUM_ANO_MES = 202205 and (QTD_TRANSACAO) > 0 then 1 else 0 end as FLG_202205
    ,case when base.NUM_ANO_MES = 202206 and (QTD_TRANSACAO) > 0 then 1 else 0 end as FLG_202206
    from
        (
            select
            ACCOUNT_ID -- Checking account unique identification number
            CREDIT_UNION_ID -- Credit Union ID Number
            ACCOUNT_NUM -- Checking Account Number
            ,CAST(TO_CHAR(ht.dat_data_proc,'yyyymm') AS INT) as NUM_YEAR_MONTH -- Date in YYYYMM format (integer) 
            ,case
            when TRANSACTION_CODE in (X,Y,Z) then 'Channels'
            when TRANSACTION_CODE in (A,B,C) then 'Credit & Debit Cards'
            when TRANSACTION_CODE in (1,2,3) then 'Checking Account'
            when TRANSACTION_CODE in (D,E,F) then then 'Bills'
            when TRANSACTION_CODE in (0,9,8) then 'Credit'
            when TRANSACTION_CODE in (P,Q,R) then 'Investiments'
            when TRANSACTION_CODE in (M,N,O) then 'Payments'
            else 'N/A' end as PRODUCT -- Product name
            ,count(1) as NUM_TRANSACTIONS -- Counts the number of transactions
            from MEMBER_TRANSACTIONS
            where 1=1 
            and CHANNEL = 'Mobile App'-- Filter to bring up only Mobile app usage
            and SITUATION = 'Completed' -- Filter to bring only sucefull transactions
            and CREDIT_UNION_ID = 'A'  -- Filter to bring only credit unions of interest
            and INDIVIDUAL_OR_COMPANY = 'Individual' -- Filter to bring only individuals
            and TRANSACTION_DATE between timestamp '2022-01-01 00:00:00' and timestamp '2022-06-30 23:59:59' -- Filter to bring only transactions in the first semester of 2022
            and TRANSACTION_CODE in (...) -- Filter to only bring financial transactions
            group by 1,2,3,4,5     
            ) base

;