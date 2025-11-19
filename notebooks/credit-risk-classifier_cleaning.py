
# Etapa 1: Limpeza dos dados

#%% Import das bibliotecas
import pandas as pd
import numpy as np
import re

#%%  Leitura dos arquivos
arquivo_test = pd.read_csv('test.csv')
arquivo_train = pd.read_csv('train.csv')

# Junção dos arquivos
df = pd.concat([arquivo_test,arquivo_train], ignore_index=True)
df

#%%  FUNÇÕES GERAIS

def clean_simple_text_column(df, col, id_col='Customer_ID', invalid_values=None):
    """
    Função para limpar colunas de texto simples.
    Remove valores inválidos e preenche com último valor válido do cliente.
    """
    s = df[col].astype('string')
    
    if invalid_values:
        for val in invalid_values:
            s = s.replace(val, pd.NA)
    
    s_filled = s.groupby(df[id_col]).transform('last')
    df[col] = s_filled
    
    return df


def clean_numeric_with_mode(df, col, id_col='Customer_ID', invalid_patterns=None, remove_underscores=True):
    """
    Função para limpar colunas numéricas usando moda por cliente.
    - Remove padrões inválidos
    - Calcula moda por cliente
    - Substitui valores discrepantes
    - Preenche NAs com último valor válido
    """
    # Limpeza inicial
    s = df[col].astype('string').str.strip()
    
    if remove_underscores:
        s = s.str.replace('_', '', regex=False)
    
    if invalid_patterns:
        for pattern in invalid_patterns:
            s = s.replace(pattern, pd.NA)
    
    df[col] = pd.to_numeric(s, errors='coerce')
    
    # Moda por cliente
    mode_per_id = df.groupby(id_col)[col].transform( lambda x: x.mode(dropna=True).iloc[0] if not x.mode(dropna=True).empty else pd.NA)
    
    # Valores discrepantes
    multi_vals = df.groupby(id_col)[col].transform(lambda x: x.dropna().nunique() > 1)
    mask_diff = multi_vals & df[col].notna() & (df[col] != mode_per_id)
    df.loc[mask_diff, col] = pd.NA
    
    # Preenche NAs
    last_valid = df.groupby(id_col)[col].transform(lambda x: x.dropna().iloc[-1] if x.dropna().size else pd.NA)
    df.loc[:, col] = df[col].fillna(last_valid)
    return df


def clean_numeric_with_threshold(df, col, id_col='Customer_ID', thresh=20):
    """
    Função para limpar colunas numéricas com limiar de diferença.
    Remove valores que diferem muito da moda do cliente.
    """
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Moda por cliente
    mode_per_id = df.groupby(id_col)[col].transform( lambda s: s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else pd.NA)
    
    # Marca diferenças >= thresh
    diff = (df[col] - mode_per_id).abs()
    mask = df[col].notna() & mode_per_id.notna() & (diff >= thresh)
    
    # Aplica NA e preenche com moda
    df.loc[mask, col] = pd.NA
    df.loc[:, col] = df[col].where(df[col].notna(), mode_per_id)
    
    return df


def clean_categorical_with_mode(df, col, id_col='Customer_ID',invalid_values=None, valid_set=None):
    """
    Função para limpar colunas categóricas usando moda.
    """
    s = df[col].astype('string').str.strip()
    
    if invalid_values:
        for val in invalid_values:
            s = s.replace(val, pd.NA)
    
    df.loc[:, col] = s
    
    # Moda por cliente 
    def mode_func(x):
        if valid_set:
            x = x[x.isin(valid_set)]
        m = x.mode(dropna=True)
        return m.iloc[0] if not m.empty else pd.NA
    
    mode_per_id = df.groupby(id_col)[col].transform(mode_func)
    
    # Preenche NAs
    na_mask = df[col].isna()
    df.loc[na_mask, col] = mode_per_id[na_mask]
    
    df.loc[:, col] = df[col].astype('string')
    return df


def clean_amount_with_median(df, col, id_col='Customer_ID', special_tokens=None, remove_underscores=True):
    """
    Função para limpar valores monetários usando mediana.
    """
    s = df[col].astype('string').str.strip()
    
    if special_tokens:
        s = s.mask(s.isin(special_tokens), pd.NA)
    
    if remove_underscores:
        s = s.str.replace('_', '', regex=False)
    
    s_num = pd.to_numeric(s, errors='coerce')
    df[col] = s_num
    
    # Preenche com mediana por cliente
    med_id = df.groupby(id_col)[col].transform('median')
    na_mask = df[col].isna()
    df.loc[na_mask, col] = med_id[na_mask]
    
    return df

# FUNÇÕES ESPECÍFICAS 

def format_age(df, col='Age', min_age=0, max_age=99):
    """Formata a coluna Idade"""
    s = df[col].astype('string').str.strip()
    s = s.str.replace('_','',regex=False)
    
    mask_long = s.str.len() > 3 
    mask_neg = s.str.contains('-',na=False)
    s = s.mask(mask_long | mask_neg)
    
    s_num = pd.to_numeric(s, errors='coerce')    
    out_of_range = (s_num < min_age) | (s_num > max_age)
    s_num = s_num.mask(out_of_range)
    
    df[col] = s_num.astype('Int64')
    return df


def retriver_missing_age(df, col='Age'):

    """Recupera idades ausentes baseado em Customer_ID e Month"""

    df[col] = pd.to_numeric(df[col], errors='coerce')
    month_norm = df['Month'].astype('string').str.strip().str.title()
    
    month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12 }
    
    df['Month_Num'] = month_norm.map(month_map)
    df = df.sort_values(['Customer_ID', 'Month_Num'], kind='mergesort')
    
    df[col] = (df.groupby('Customer_ID')[col]
                 .apply(lambda s: s.ffill().bfill())
                 .reset_index(level=0, drop=True))
    
    col = df[col].astype('Int64')
    df = df.drop(columns=['Month_Num'])
    
    return df


def fill_missing_values(df):
    """Preenche valores de idade ausentes com 34"""
    df['Age'] = df['Age'].fillna(34)
    return df


def fix_age_for_ids(df, customer_id_outlier, id_col='Customer_ID', col='Age'):
    """Corrige outliers de idade para IDs específicos"""
    mfix = df[id_col].isin(customer_id_outlier)
    
    modes = (df.loc[mfix]
               .groupby(id_col)[col]
               .agg(lambda x: x.mode(dropna=True).iloc[0] if not x.mode(dropna=True).empty else pd.NA))
    
    moda_linha = df[id_col].map(modes)
    mask_sub = mfix & df[col].notna() & moda_linha.notna() & df[col].ne(moda_linha)
    df.loc[mask_sub, col] = moda_linha[mask_sub]
   
    return df


def parse_credit_history_to_months(df, src_col='Credit_History_Age', dst_col='Credit_History_Age_Months', months_per_year=12):
    """Converte texto de histórico de crédito para meses"""
    
    RE_YEARS = re.compile(r'(\d+)\s*(?:year|years|yr|yrs|y)\b')
    RE_MONTHS = re.compile(r'(\d+)\s*(?:month|months|mo|mos|m)\b')
    RE_YEAR_MONTH_SEP = re.compile(r'^\s*(\d+)\s*[/\-]\s*(\d+)\s*$')
    RE_DECIMAL_YEARS = re.compile(r'^\d+(\.\d+)?$')
    RE_WHITESPACE = re.compile(r'\s+')
    
    def _to_months(val):
        if pd.isna(val):
            return pd.NA
        
        s = str(val).strip().lower() \
            .replace('_', ' ') \
            .replace(',', ' ') \
            .replace('+', ' ')
        
        s = RE_WHITESPACE.sub(' ', s)
        
        y = RE_YEARS.search(s)
        m = RE_MONTHS.search(s)
        if y or m:
            yy = int(y.group(1)) if y else 0
            mm = int(m.group(1)) if m else 0
            return yy * months_per_year + mm
        
        m2 = RE_YEAR_MONTH_SEP.match(s)
        if m2:
            return int(m2.group(1)) * months_per_year + int(m2.group(2))
        
        if RE_DECIMAL_YEARS.match(s):
            return int(round(float(s) * months_per_year))
        
        return pd.NA
    
    df[dst_col] = df[src_col].apply(_to_months)
    return df


def fill_sequential_months(df, fill_col='Credit_History_Age_Months', id_col='Customer_ID', month_col='Month'):
    """Preenche meses de forma sequencial"""
    
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12,
        'Other': 13
    }
    df['__Month_Num__'] = df[month_col].str.title().map(month_map)
    df['__orig__'] = np.arange(len(df))
    
    df = df.sort_values([id_col, '__Month_Num__', '__orig__'], kind='mergesort')
    
    def _sequential_increment(s):
        base = s.ffill()
        subgroup_id = s.notna().cumsum()
        increment = s.groupby(subgroup_id).cumcount()
        s_filled = (base + increment)
        s_filled = s_filled.bfill()
        return s_filled
    
    df[fill_col] = df.groupby(id_col)[fill_col].transform(_sequential_increment)
    df = df.sort_values('__orig__').drop(columns=['__orig__', '__Month_Num__'])
    
    return df

def fix_delay_from_due_date(df,delay_col='Delay_from_due_date',loan_col='Num_of_Loan', id_col='Customer_ID'):
    
    df.loc[:, delay_col] = pd.to_numeric(df[delay_col], errors='coerce')

    # Para valores menores do que 0 substituir por NA
    df.loc[df[delay_col] < 0, delay_col] = pd.NA

    # Para valores em que não há empréstimo subsituir por 0
    no_loan_mask = df[loan_col] == 0
    df.loc[no_loan_mask, delay_col] = 0

    # Para valores NA, pegar o maior valor dentro do Customer_ID
    grp_max = df.groupby(id_col)[delay_col].transform('max')
    na_mask = df[delay_col].isna()
    df.loc[na_mask, delay_col] = grp_max[na_mask]  

    return df

#%%  APLICAÇÃO DAS FUNÇÕES

# Age 
df = format_age(df, col='Age')
df = retriver_missing_age(df, col='Age')
df = fill_missing_values(df)

# Valores identificados como outliers (customer_id_outlier)
customer_id_outlier = ["CUS_0x1dd3", "CUS_0x32fa", "CUS_0x6423", "CUS_0x7e94", "CUS_0xc24a", "CUS_0xdc8"]
df = fix_age_for_ids(df, customer_id_outlier, id_col='Customer_ID', col='Age')

# Colunas de texto simples
df = clean_simple_text_column(df, 'Occupation', invalid_values=['_______'])
df = clean_simple_text_column(df, 'Name')

# SSN 
df['SSN'] = df['SSN'].astype('string').replace('#F%$D@*&8', pd.NA)
df['SSN'] = df['SSN'].str.replace(r'\D', '', regex=True)
df['SSN'] = pd.to_numeric(df['SSN'], errors='coerce')
df['SSN'] = df.groupby('Customer_ID')['SSN'].transform('last')
# Colunas numéricas com moda
df = clean_numeric_with_mode(df, 'Num_Bank_Accounts')
df = clean_numeric_with_mode(df, 'Num_Credit_Card')
df = clean_numeric_with_mode(df, 'Interest_Rate')
df = clean_numeric_with_mode(df, 'Num_of_Loan')
df = clean_numeric_with_mode(df, 'Num_of_Delayed_Payment')

# Annual Income e Monthly Salary 
df = clean_numeric_with_mode(df, 'Annual_Income')
df = clean_numeric_with_mode(df, 'Monthly_Inhand_Salary')
df['Monthly_Inhand_Salary'] = df['Monthly_Inhand_Salary'].round(2)

# Colunas com threshold
df = clean_numeric_with_threshold(df, 'Num_Credit_Inquiries', thresh=20)
df = clean_numeric_with_threshold(df, 'Total_EMI_per_month', thresh=100)

# Outstanding_Debt 
df['Outstanding_Debt'] = df['Outstanding_Debt'].astype('string').str.replace('_','',regex=False)

# Changed_Credit_Limit 
s = df['Changed_Credit_Limit'].astype('string').str.strip().str.replace('_', '', regex=False)
df['Changed_Credit_Limit'] = pd.to_numeric(s, errors='coerce').round(2).astype('Float64')

month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
df['Month_Num'] = df['Month'].str.title().map(month_map)
df = df.sort_values(['Customer_ID', 'Month_Num'], kind='mergesort')
df['Changed_Credit_Limit'] = (df.groupby('Customer_ID')['Changed_Credit_Limit']
                               .apply(lambda s: s.ffill().bfill())
                               .reset_index(level=0, drop=True))
df = df.drop(columns=['Month_Num'])

# Type_of_Loan
df['Type_of_Loan'] = df['Type_of_Loan'].fillna('No Loan').astype('string')

# Colunas categóricas com moda
df = clean_categorical_with_mode(df, 'Credit_Mix', invalid_values=['_'])
df = clean_categorical_with_mode(df, 'Payment_Behaviour', invalid_values=['!@9#%8'])
df = clean_categorical_with_mode(df, 'Credit_Score')

# Payment_of_Min_Amount 
s = df['Payment_of_Min_Amount'].astype('string').str.strip().str.upper()
s = s.replace({'Y':'YES', 'N':'NO'})
df['Payment_of_Min_Amount'] = s

def mode_yes_no(x):
    m = x[x.isin(['YES','NO'])].mode()
    return m.iloc[0] if not m.empty else pd.NA

mode_per_id = df.groupby('Customer_ID')['Payment_of_Min_Amount'].transform(mode_yes_no)
mask_nm = df['Payment_of_Min_Amount'].isin(['NM'])
df.loc[mask_nm, 'Payment_of_Min_Amount'] = mode_per_id[mask_nm]

# Amount_invested_monthly e Monthly_Balance 
df = clean_amount_with_median(df, 'Amount_invested_monthly', special_tokens={'__10000__'})
df = clean_amount_with_median(df, 'Monthly_Balance', remove_underscores=False)

# Credit_History_Age 
df = parse_credit_history_to_months(df, src_col='Credit_History_Age', dst_col='Credit_History_Age_Months')
df = fill_sequential_months(df, fill_col='Credit_History_Age_Months', id_col='Customer_ID', month_col='Month')

# Delay_from_due_date
df = fix_delay_from_due_date(df,delay_col='Delay_from_due_date',loan_col='Num_of_Loan', id_col='Customer_ID')

# Conversão de type
conv_type_float = ['Annual_Income','Amount_invested_monthly','Monthly_Balance','Credit_Utilization_Ratio','Total_EMI_per_month'] 
conv_type_int = ['Num_of_Loan','Num_of_Delayed_Payment','Delay_from_due_date']

dtype_map = {
    **{col: 'Float64' for col in conv_type_float},
    **{col: 'Int32' for col in conv_type_int}
}

df = df.astype(dtype_map)
df[conv_type_float] = df[conv_type_float].round(3)

# %% SALVANDO ARQUIVO
df.to_csv('credit_score_classification.csv', index=False, encoding='utf-8')


