/*

This is a pseudo SQL Algorithm built to show the basic data wrangling used to prepare the data for model development and evaluation.

*/

select
-- Identification
flg_outter.ACCOUNT_ID -- Checking account unique identification number
,flg_outter.CREDIT_UNION_ID -- Credit Union ID number
,flg_outter.ACCOUNT_NUM -- Checking account number
-- Churn flags
,case when flg_outter.FLG_202201 > 0 then 1 else 0 end as FLG_202201
,case when flg_outter.FLG_202202 > 0 then 1 else 0 end as FLG_202202
,case when flg_outter.FLG_202203 > 0 then 1 else 0 end as FLG_202203
,case when flg_outter.FLG_202204 > 0 then 1 else 0 end as FLG_202204
,case when flg_outter.FLG_202205 > 0 then 1 else 0 end as FLG_202205
,case when flg_outter.FLG_202206 > 0 then 1 else 0 end as FLG_202206
-- Deepness :: Channels
,flg_outter.DEEP_CHANNELS_202201
,flg_outter.DEEP_CHANNELS_202202
,flg_outter.DEEP_CHANNELS_202203
,flg_outter.DEEP_CHANNELS_202204
,flg_outter.DEEP_CHANNELS_202205
,flg_outter.DEEP_CHANNELS_202206
-- Deepness :: Credit & Debit Cards
,flg_outter.DEEP_CARDS_202201
,flg_outter.DEEP_CARDS_202202
,flg_outter.DEEP_CARDS_202203
,flg_outter.DEEP_CARDS_202204
,flg_outter.DEEP_CARDS_202205
,flg_outter.DEEP_CARDS_202206
-- Deepness :: Checking Account
,flg_outter.DEEP_CHECKING_202201
,flg_outter.DEEP_CHECKING_202202
,flg_outter.DEEP_CHECKING_202203
,flg_outter.DEEP_CHECKING_202204
,flg_outter.DEEP_CHECKING_202205
,flg_outter.DEEP_CHECKING_202206
-- Deepness :: Bills
,flg_outter.DEEP_BILLS_202201
,flg_outter.DEEP_BILLS_202202
,flg_outter.DEEP_BILLS_202203
,flg_outter.DEEP_BILLS_202204
,flg_outter.DEEP_BILLS_202205
,flg_outter.DEEP_BILLS_202206
-- Deepness :: Credit
,flg_outter.DEEP_CREDIT_202201
,flg_outter.DEEP_CREDIT_202202
,flg_outter.DEEP_CREDIT_202203
,flg_outter.DEEP_CREDIT_202204
,flg_outter.DEEP_CREDIT_202205
,flg_outter.DEEP_CREDIT_202206
-- Deepness :: Investiments
,flg_outter.DEEP_INVESTMENTS_202201
,flg_outter.DEEP_INVESTMENTS_202202
,flg_outter.DEEP_INVESTMENTS_202203
,flg_outter.DEEP_INVESTMENTS_202204
,flg_outter.DEEP_INVESTMENTS_202205
,flg_outter.DEEP_INVESTMENTS_202206
-- Deepness :: Payments
,flg_outter.DEEP_PAYMENTS_202201
,flg_outter.DEEP_PAYMENTS_202202
,flg_outter.DEEP_PAYMENTS_202203
,flg_outter.DEEP_PAYMENTS_202204
,flg_outter.DEEP_PAYMENTS_202205
,flg_outter.DEEP_PAYMENTS_202206
-- Deepness :: PIX
,flg_outter.DEEP_PIX_202201
,flg_outter.DEEP_PIX_202202
,flg_outter.DEEP_PIX_202203
,flg_outter.DEEP_PIX_202204
,flg_outter.DEEP_PIX_202205
,flg_outter.DEEP_PIX_202206
-- Amplitude (Number of products used)
,sum(
    case when flg_outter.DEEP_CHANNELS_202201 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_CARDS_202201 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_CHECKING_202201 >0 then 1 else 0 end +
    case when flg_outter.DEEP_BILLS_202201 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_CREDIT_202201 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_INVESTMENTS_202201 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_PAYMENTS_202201 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_PIX_202201 > 0 then 1 else 0 end 
    ) as AMP_202201 
,sum(
    case when flg_outter.DEEP_CHANNELS_202202 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_CARDS_202202 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_CHECKING_202202 >0 then 1 else 0 end +
    case when flg_outter.DEEP_BILLS_202202 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_CREDIT_202202 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_INVESTMENTS_202202 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_PAYMENTS_202202 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_PIX_202202 > 0 then 1 else 0 end 
    ) as AMP_202202
,sum(
    case when flg_outter.DEEP_CHANNELS_202203 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_CARDS_202203 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_CHECKING_202203 >0 then 1 else 0 end +
    case when flg_outter.DEEP_BILLS_202203 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_CREDIT_202203 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_INVESTMENTS_202203 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_PAYMENTS_202203 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_PIX_202203 > 0 then 1 else 0 end 
    ) as AMP_202203
,sum(
    case when flg_outter.DEEP_CHANNELS_202204 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_CARDS_202204 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_CHECKING_202204 >0 then 1 else 0 end +
    case when flg_outter.DEEP_BILLS_202204 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_CREDIT_202204 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_INVESTMENTS_202204 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_PAYMENTS_202204 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_PIX_202204 > 0 then 1 else 0 end 
    ) as AMP_202204
,sum(
    case when flg_outter.DEEP_CHANNELS_202205 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_CARDS_202205 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_CHECKING_202205 >0 then 1 else 0 end +
    case when flg_outter.DEEP_BILLS_202205 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_CREDIT_202205 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_INVESTMENTS_202205 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_PAYMENTS_202205 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_PIX_202205 > 0 then 1 else 0 end 
    ) as AMP_202205
,sum(
    case when flg_outter.DEEP_CHANNELS_202206 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_CARDS_202206 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_CHECKING_202206 >0 then 1 else 0 end +
    case when flg_outter.DEEP_BILLS_202206 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_CREDIT_202206 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_INVESTMENTS_202206 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_PAYMENTS_202206 > 0 then 1 else 0 end +
    case when flg_outter.DEEP_PIX_202206 > 0 then 1 else 0 end 
    ) as AMP_202206
-- Number of transactions
,sum(
    flg_outter.DEEP_CHANNELS_202201 +
    flg_outter.DEEP_CARDS_202201 +
    flg_outter.DEEP_CHECKING_202201 +
    flg_outter.DEEP_BILLS_202201 +
    flg_outter.DEEP_CREDIT_202201 +
    flg_outter.DEEP_INVESTMENTS_202201 +
    flg_outter.DEEP_PAYMENTS_202201 +
    flg_outter.DEEP_PIX_202201
    ) as NUM_TRANSACTIONS_202201
,sum(
    flg_outter.DEEP_CHANNELS_202202 +
    flg_outter.DEEP_CARDS_202202 +
    flg_outter.DEEP_CHECKING_202202 +
    flg_outter.DEEP_BILLS_202202 +
    flg_outter.DEEP_CREDIT_202202 +
    flg_outter.DEEP_INVESTMENTS_202202 +
    flg_outter.DEEP_PAYMENTS_202202 +
    flg_outter.DEEP_PIX_202202
    ) as NUM_TRANSACTIONS_202202
,sum(
    flg_outter.DEEP_CHANNELS_202203 +
    flg_outter.DEEP_CARDS_202203 +
    flg_outter.DEEP_CHECKING_202203 +
    flg_outter.DEEP_BILLS_202203 +
    flg_outter.DEEP_CREDIT_202203 +
    flg_outter.DEEP_INVESTMENTS_202203 +
    flg_outter.DEEP_PAYMENTS_202203 +
    flg_outter.DEEP_PIX_202203
    ) as NUM_TRANSACTIONS_202203
,sum(
    flg_outter.DEEP_CHANNELS_202204 +
    flg_outter.DEEP_CARDS_202204 +
    flg_outter.DEEP_CHECKING_202204 +
    flg_outter.DEEP_BILLS_202204 +
    flg_outter.DEEP_CREDIT_202204 +
    flg_outter.DEEP_INVESTMENTS_202204 +
    flg_outter.DEEP_PAYMENTS_202204 +
    flg_outter.DEEP_PIX_202204
    ) as NUM_TRANSACTIONS_202204
,sum(
    flg_outter.DEEP_CHANNELS_202205 +
    flg_outter.DEEP_CARDS_202205 +
    flg_outter.DEEP_CHECKING_202205 +
    flg_outter.DEEP_BILLS_202205 +
    flg_outter.DEEP_CREDIT_202205 +
    flg_outter.DEEP_INVESTMENTS_202205 +
    flg_outter.DEEP_PAYMENTS_202205 +
    flg_outter.DEEP_PIX_202205
    ) as NUM_TRANSACTIONS_202205
,sum(
    flg_outter.DEEP_CHANNELS_202206 +
    flg_outter.DEEP_CARDS_202206 +
    flg_outter.DEEP_CHECKING_202206 +
    flg_outter.DEEP_BILLS_202206 +
    flg_outter.DEEP_CREDIT_202206 +
    flg_outter.DEEP_INVESTMENTS_202206 +
    flg_outter.DEEP_PAYMENTS_202206 +
    flg_outter.DEEP_PIX_202206
    ) as NUM_TRANSACTIONS_202206
from
(
    select 
    flg_inner.ACCOUNT_ID -- Checking account identification
    ,flg_inner.CREDIT_UNION_ID -- Credit Union ID Number
    ,flg_inner.ACCOUNT_NUM -- Checking account number
    -- Churn flags
    ,sum(flg_inner.FLG_202201) as FLG_202201 
    ,sum(flg_inner.FLG_202202) as FLG_202202
    ,sum(flg_inner.FLG_202203) as FLG_202203
    ,sum(flg_inner.FLG_202204) as FLG_202204
    ,sum(flg_inner.FLG_202205) as FLG_202205
    ,sum(flg_inner.FLG_202206) as FLG_202206
    -- Deepness :: Channels
    ,sum(flg_inner.DEEP_CHANNELS_202201) as DEEP_CHANNELS_202201
    ,sum(flg_inner.DEEP_CHANNELS_202202) as DEEP_CHANNELS_202202
    ,sum(flg_inner.DEEP_CHANNELS_202203) as DEEP_CHANNELS_202203
    ,sum(flg_inner.DEEP_CHANNELS_202204) as DEEP_CHANNELS_202204
    ,sum(flg_inner.DEEP_CHANNELS_202205) as DEEP_CHANNELS_202205
    ,sum(flg_inner.DEEP_CHANNELS_202206) as DEEP_CHANNELS_202206
    -- Deepness :: Credit & Debit Cards
    ,sum(flg_inner.DEEP_CARDS_202201) as DEEP_CARDS_202201
    ,sum(flg_inner.DEEP_CARDS_202202) as DEEP_CARDS_202202
    ,sum(flg_inner.DEEP_CARDS_202203) as DEEP_CARDS_202203
    ,sum(flg_inner.DEEP_CARDS_202204) as DEEP_CARDS_202204
    ,sum(flg_inner.DEEP_CARDS_202205) as DEEP_CARDS_202205
    ,sum(flg_inner.DEEP_CARDS_202206) as DEEP_CARDS_202206
    -- Deepness :: Checking Account
    ,sum(flg_inner.DEEP_CHECKING_202201) as DEEP_CHECKING_202201
    ,sum(flg_inner.DEEP_CHECKING_202202) as DEEP_CHECKING_202202
    ,sum(flg_inner.DEEP_CHECKING_202203) as DEEP_CHECKING_202203
    ,sum(flg_inner.DEEP_CHECKING_202204) as DEEP_CHECKING_202204
    ,sum(flg_inner.DEEP_CHECKING_202205) as DEEP_CHECKING_202205
    ,sum(flg_inner.DEEP_CHECKING_202206) as DEEP_CHECKING_202206
    -- Deepness :: Bills
    ,sum(flg_inner.DEEP_BILLS_202201) as DEEP_BILLS_202201
    ,sum(flg_inner.DEEP_BILLS_202202) as DEEP_BILLS_202202
    ,sum(flg_inner.DEEP_BILLS_202203) as DEEP_BILLS_202203
    ,sum(flg_inner.DEEP_BILLS_202204) as DEEP_BILLS_202204
    ,sum(flg_inner.DEEP_BILLS_202205) as DEEP_BILLS_202205
    ,sum(flg_inner.DEEP_BILLS_202206) as DEEP_BILLS_202206
    -- Deepness :: Credit
    ,sum(flg_inner.DEEP_CREDIT_202201) as DEEP_CREDIT_202201
    ,sum(flg_inner.DEEP_CREDIT_202202) as DEEP_CREDIT_202202
    ,sum(flg_inner.DEEP_CREDIT_202203) as DEEP_CREDIT_202203
    ,sum(flg_inner.DEEP_CREDIT_202204) as DEEP_CREDIT_202204
    ,sum(flg_inner.DEEP_CREDIT_202205) as DEEP_CREDIT_202205
    ,sum(flg_inner.DEEP_CREDIT_202206) as DEEP_CREDIT_202206
    -- Deepness :: Investiments
    ,sum(flg_inner.DEEP_INVESTMENTS_202201) as DEEP_INVESTMENTS_202201
    ,sum(flg_inner.DEEP_INVESTMENTS_202202) as DEEP_INVESTMENTS_202202
    ,sum(flg_inner.DEEP_INVESTMENTS_202203) as DEEP_INVESTMENTS_202203
    ,sum(flg_inner.DEEP_INVESTMENTS_202204) as DEEP_INVESTMENTS_202204
    ,sum(flg_inner.DEEP_INVESTMENTS_202205) as DEEP_INVESTMENTS_202205
    ,sum(flg_inner.DEEP_INVESTMENTS_202206) as DEEP_INVESTMENTS_202206
    -- Deepness :: Payments
    ,sum(flg_inner.DEEP_PAYMENTS_202201) as DEEP_PAYMENTS_202201
    ,sum(flg_inner.DEEP_PAYMENTS_202202) as DEEP_PAYMENTS_202202
    ,sum(flg_inner.DEEP_PAYMENTS_202203) as DEEP_PAYMENTS_202203
    ,sum(flg_inner.DEEP_PAYMENTS_202204) as DEEP_PAYMENTS_202204
    ,sum(flg_inner.DEEP_PAYMENTS_202205) as DEEP_PAYMENTS_202205
    ,sum(flg_inner.DEEP_PAYMENTS_202206) as DEEP_PAYMENTS_202206
    -- Deepness :: PIX
    ,sum(flg_inner.DEEP_PIX_202201) as DEEP_PIX_202201
    ,sum(flg_inner.DEEP_PIX_202202) as DEEP_PIX_202202
    ,sum(flg_inner.DEEP_PIX_202203) as DEEP_PIX_202203
    ,sum(flg_inner.DEEP_PIX_202204) as DEEP_PIX_202204
    ,sum(flg_inner.DEEP_PIX_202205) as DEEP_PIX_202205
    ,sum(flg_inner.DEEP_PIX_202206) as DEEP_PIX_202206
    from
    (
        select 
        base.ACCOUNT_ID -- Checking account identification
        ,base.CREDIT_UNION_ID -- Credit Union ID Number
        ,base.ACCOUNT_NUM -- Checking account number
        -- Churn flags
        ,case when base.NUM_YEAR_MONTH = 202201 and (base.NUM_TRANSACTIONS) > 0 then 1 else 0 end as FLG_202201
        ,case when base.NUM_YEAR_MONTH = 202202 and (base.NUM_TRANSACTIONS) > 0 then 1 else 0 end as FLG_202202
        ,case when base.NUM_YEAR_MONTH = 202203 and (base.NUM_TRANSACTIONS) > 0 then 1 else 0 end as FLG_202203
        ,case when base.NUM_YEAR_MONTH = 202204 and (base.NUM_TRANSACTIONS) > 0 then 1 else 0 end as FLG_202204
        ,case when base.NUM_YEAR_MONTH = 202205 and (base.NUM_TRANSACTIONS) > 0 then 1 else 0 end as FLG_202205
        ,case when base.NUM_YEAR_MONTH = 202206 and (base.NUM_TRANSACTIONS) > 0 then 1 else 0 end as FLG_202206
        -- Deepness :: Channels
        ,case when base.NUM_YEAR_MONTH = 202201 and base.PRODUCT = 'Channels' then base.NUM_TRANSACTIONS else 0 end as DEEP_CHANNELS_202201
        ,case when base.NUM_YEAR_MONTH = 202202 and base.PRODUCT = 'Channels' then base.NUM_TRANSACTIONS else 0 end as DEEP_CHANNELS_202202
        ,case when base.NUM_YEAR_MONTH = 202203 and base.PRODUCT = 'Channels' then base.NUM_TRANSACTIONS else 0 end as DEEP_CHANNELS_202203
        ,case when base.NUM_YEAR_MONTH = 202204 and base.PRODUCT = 'Channels' then base.NUM_TRANSACTIONS else 0 end as DEEP_CHANNELS_202204
        ,case when base.NUM_YEAR_MONTH = 202205 and base.PRODUCT = 'Channels' then base.NUM_TRANSACTIONS else 0 end as DEEP_CHANNELS_202205
        ,case when base.NUM_YEAR_MONTH = 202206 and base.PRODUCT = 'Channels' then base.NUM_TRANSACTIONS else 0 end as DEEP_CHANNELS_202206
        -- Deepness :: Credit & Debit Cards
        ,case when base.NUM_YEAR_MONTH = 202201 and base.PRODUCT = 'Credit & Debit Cards' then base.NUM_TRANSACTIONS else 0 end as DEEP_CARDS_202201
        ,case when base.NUM_YEAR_MONTH = 202202 and base.PRODUCT = 'Credit & Debit Cards' then base.NUM_TRANSACTIONS else 0 end as DEEP_CARDS_202202
        ,case when base.NUM_YEAR_MONTH = 202203 and base.PRODUCT = 'Credit & Debit Cards' then base.NUM_TRANSACTIONS else 0 end as DEEP_CARDS_202203
        ,case when base.NUM_YEAR_MONTH = 202204 and base.PRODUCT = 'Credit & Debit Cards' then base.NUM_TRANSACTIONS else 0 end as DEEP_CARDS_202204
        ,case when base.NUM_YEAR_MONTH = 202205 and base.PRODUCT = 'Credit & Debit Cards' then base.NUM_TRANSACTIONS else 0 end as DEEP_CARDS_202205
        ,case when base.NUM_YEAR_MONTH = 202206 and base.PRODUCT = 'Credit & Debit Cards' then base.NUM_TRANSACTIONS else 0 end as DEEP_CARDS_202206
        -- Deepness :: Checking Account
        ,case when base.NUM_YEAR_MONTH = 202201 and base.PRODUCT = 'Checking Account' then base.NUM_TRANSACTIONS else 0 end as DEEP_CHECKING_202201
        ,case when base.NUM_YEAR_MONTH = 202202 and base.PRODUCT = 'Checking Account' then base.NUM_TRANSACTIONS else 0 end as DEEP_CHECKING_202202
        ,case when base.NUM_YEAR_MONTH = 202203 and base.PRODUCT = 'Checking Account' then base.NUM_TRANSACTIONS else 0 end as DEEP_CHECKING_202203
        ,case when base.NUM_YEAR_MONTH = 202204 and base.PRODUCT = 'Checking Account' then base.NUM_TRANSACTIONS else 0 end as DEEP_CHECKING_202204
        ,case when base.NUM_YEAR_MONTH = 202205 and base.PRODUCT = 'Checking Account' then base.NUM_TRANSACTIONS else 0 end as DEEP_CHECKING_202205
        ,case when base.NUM_YEAR_MONTH = 202206 and base.PRODUCT = 'Checking Account' then base.NUM_TRANSACTIONS else 0 end as DEEP_CHECKING_202206
        -- Deepness :: Bills
        ,case when base.NUM_YEAR_MONTH = 202201 and base.PRODUCT = 'Bills' then base.NUM_TRANSACTIONS else 0 end as DEEP_BILLS_202201
        ,case when base.NUM_YEAR_MONTH = 202202 and base.PRODUCT = 'Bills' then base.NUM_TRANSACTIONS else 0 end as DEEP_BILLS_202202
        ,case when base.NUM_YEAR_MONTH = 202203 and base.PRODUCT = 'Bills' then base.NUM_TRANSACTIONS else 0 end as DEEP_BILLS_202203
        ,case when base.NUM_YEAR_MONTH = 202204 and base.PRODUCT = 'Bills' then base.NUM_TRANSACTIONS else 0 end as DEEP_BILLS_202204
        ,case when base.NUM_YEAR_MONTH = 202205 and base.PRODUCT = 'Bills' then base.NUM_TRANSACTIONS else 0 end as DEEP_BILLS_202205
        ,case when base.NUM_YEAR_MONTH = 202206 and base.PRODUCT = 'Bills' then base.NUM_TRANSACTIONS else 0 end as DEEP_BILLS_202206
        -- Deepness :: Credit
        ,case when base.NUM_YEAR_MONTH = 202201 and base.PRODUCT = 'Credit' then NUM_TRANSACTIONS else 0 end as DEEP_CREDIT_202201
        ,case when base.NUM_YEAR_MONTH = 202202 and base.PRODUCT = 'Credit' then NUM_TRANSACTIONS else 0 end as DEEP_CREDIT_202202
        ,case when base.NUM_YEAR_MONTH = 202203 and base.PRODUCT = 'Credit' then NUM_TRANSACTIONS else 0 end as DEEP_CREDIT_202203
        ,case when base.NUM_YEAR_MONTH = 202204 and base.PRODUCT = 'Credit' then NUM_TRANSACTIONS else 0 end as DEEP_CREDIT_202204
        ,case when base.NUM_YEAR_MONTH = 202205 and base.PRODUCT = 'Credit' then NUM_TRANSACTIONS else 0 end as DEEP_CREDIT_202205
        ,case when base.NUM_YEAR_MONTH = 202206 and base.PRODUCT = 'Credit' then NUM_TRANSACTIONS else 0 end as DEEP_CREDIT_202206
        -- Deepness :: Investiments
        ,case when base.NUM_YEAR_MONTH = 202201 and base.PRODUCT = 'Investiments' then base.NUM_TRANSACTIONS else 0 end as DEEP_INVESTMENTS_202201
        ,case when base.NUM_YEAR_MONTH = 202202 and base.PRODUCT = 'Investiments' then base.NUM_TRANSACTIONS else 0 end as DEEP_INVESTMENTS_202202
        ,case when base.NUM_YEAR_MONTH = 202203 and base.PRODUCT = 'Investiments' then base.NUM_TRANSACTIONS else 0 end as DEEP_INVESTMENTS_202203
        ,case when base.NUM_YEAR_MONTH = 202204 and base.PRODUCT = 'Investiments' then base.NUM_TRANSACTIONS else 0 end as DEEP_INVESTMENTS_202204
        ,case when base.NUM_YEAR_MONTH = 202205 and base.PRODUCT = 'Investiments' then base.NUM_TRANSACTIONS else 0 end as DEEP_INVESTMENTS_202205
        ,case when base.NUM_YEAR_MONTH = 202206 and base.PRODUCT = 'Investiments' then base.NUM_TRANSACTIONS else 0 end as DEEP_INVESTMENTS_202206
        -- Deepness :: Payments
        ,case when base.NUM_YEAR_MONTH = 202201 and base.PRODUCT = 'Payments' then base.NUM_TRANSACTIONS else 0 end as DEEP_PAYMENTS_202201
        ,case when base.NUM_YEAR_MONTH = 202202 and base.PRODUCT = 'Payments' then base.NUM_TRANSACTIONS else 0 end as DEEP_PAYMENTS_202202
        ,case when base.NUM_YEAR_MONTH = 202203 and base.PRODUCT = 'Payments' then base.NUM_TRANSACTIONS else 0 end as DEEP_PAYMENTS_202203
        ,case when base.NUM_YEAR_MONTH = 202204 and base.PRODUCT = 'Payments' then base.NUM_TRANSACTIONS else 0 end as DEEP_PAYMENTS_202204
        ,case when base.NUM_YEAR_MONTH = 202205 and base.PRODUCT = 'Payments' then base.NUM_TRANSACTIONS else 0 end as DEEP_PAYMENTS_202205
        ,case when base.NUM_YEAR_MONTH = 202206 and base.PRODUCT = 'Payments' then base.NUM_TRANSACTIONS else 0 end as DEEP_PAYMENTS_202206
        -- Deepness :: PIX
        ,case when base.NUM_YEAR_MONTH = 202201 and base.PRODUCT = 'PIX' then base.NUM_TRANSACTIONS else 0 end as DEEP_PIX_202201
        ,case when base.NUM_YEAR_MONTH = 202202 and base.PRODUCT = 'PIX' then base.NUM_TRANSACTIONS else 0 end as DEEP_PIX_202202
        ,case when base.NUM_YEAR_MONTH = 202203 and base.PRODUCT = 'PIX' then base.NUM_TRANSACTIONS else 0 end as DEEP_PIX_202203
        ,case when base.NUM_YEAR_MONTH = 202204 and base.PRODUCT = 'PIX' then base.NUM_TRANSACTIONS else 0 end as DEEP_PIX_202204
        ,case when base.NUM_YEAR_MONTH = 202205 and base.PRODUCT = 'PIX' then base.NUM_TRANSACTIONS else 0 end as DEEP_PIX_202205
        ,case when base.NUM_YEAR_MONTH = 202206 and base.PRODUCT = 'PIX' then base.NUM_TRANSACTIONS else 0 end as DEEP_PIX_202206
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
            ,count(ht.OID_HISTORICO) as NUM_TRANSACTIONS -- Counts the number of transactions
            from MEMBER_TRANSACTIONS
            where 1=1 
            and CHANNEL = 'Mobile App'-- Filter to bring up only Mobile app usage
            and SITUATION = 'Completed' -- Filter to bring only sucefull transactions
            and CREDIT_UNION_ID = 'A'  -- Filter to bring only credit unions of interest
            and INDIVIDUAL_OR_COMPANY = 'Individual' -- Filter to bring only individuals
            and TIMESTAMP between timestamp '2022-01-01 00:00:00' and timestamp '2022-06-30 23:59:59' -- Filter to bring only transactions in the first semester of 2022
            and TRANSACTION_CODE in (...) -- Filter to only bring financial transactions
            group by 1,2,3,4,5
           
            ) base
    ) flg_inner
    group by 1,2,3
) flg_outter
where 1=1
-- Filter only the account with movement on the prior 5 months
and flg_outter.FLG_202201 > 0
and flg_outter.FLG_202202 > 0
and flg_outter.FLG_202203 > 0 
and flg_outter.FLG_202204 > 0
and flg_outter.FLG_202205 > 0 
group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57
;
