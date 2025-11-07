import streamlit as st
import polars as pl
import numpy as np

# ============================
# SISTEMA DE LOGIN
# ============================
def check_login():
    """Verifica se o usuário está autenticado"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    return st.session_state.authenticated

def login_page():
    """Exibe a página de login"""
    st.set_page_config(page_title="Login - SIDAO", layout="centered")
    
    # Centralizar e estilizar a página de login
    st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem;
        }
        .login-title {
            text-align: center;
            color: #1f77b4;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .login-subtitle {
            text-align: center;
            color: #666;
            font-size: 1rem;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">🔐 SIDAO</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Sistema de Detecção Automática de Outliers</div>', unsafe_allow_html=True)
    
    # Formulário de login
    with st.form("login_form"):
        username = st.text_input("Usuário", placeholder="Digite seu usuário")
        password = st.text_input("Senha", type="password", placeholder="Digite sua senha")
        submit = st.form_submit_button("Entrar", use_container_width=True)
        
        if submit:
            if username == "AdminCIC" and password == "CICTCERJ25@!":
                st.session_state.authenticated = True
                st.success("✅ Login realizado com sucesso!")
                st.rerun()
            else:
                st.error("❌ Usuário ou senha incorretos")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Informações adicionais
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #999; font-size: 0.8rem;'>
            <p>Sistema desenvolvido para validação de Resultados - CIC2025</p>
            <p>Entre em contato com a CIC caso tenha esquecido sua senha</p>
        </div>
    """, unsafe_allow_html=True)

# Verificar autenticação antes de executar o script principal
if not check_login():
    login_page()
    st.stop()

# ============================
# SCRIPT PRINCIPAL (após login)
# ============================

# Função de formatação numérica

def formatar_numero(x):
    """
    Formata números no padrão brasileiro (milhar com ponto, decimal com vírgula).
    Aceita string, int ou float.
    """
    try:
        valor = float(str(x).replace('.', '').replace(',', '.'))
        if valor.is_integer():
            return f"{int(valor):,}".replace(',', '.')
        else:
            return f"{valor:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    except Exception:
        return x
import requests
from io import BytesIO

st.set_page_config(page_title="Detecção de Outliers", layout="wide")
st.title("📊 SIDAO - SISTEMA DE DETECÇÃO AUTOMÁTICA DE OUTLIERS")

# ============================
# Utils (Polars + helpers)
# ============================
def expr_to_float_br(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.replace_all(" ", "")
        .str.replace_all(r"\.", "")      # CORREÇÃO: remover '.' (milhar)
        .str.replace_all(",", ".")
        .cast(pl.Float64, strict=False)
    )

def expr_fmt_comp(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .cast(pl.Utf8, strict=False)
        .str.replace_all(r"\D+", "")
        .str.slice(0, 6)
        .cast(pl.Int64, strict=False)
    )

def prev_month_code_expr(col: str) -> pl.Expr:
    y = pl.col(col) // 100
    m = pl.col(col) % 100
    return (
        pl.when(pl.col(col).is_null())
        .then(pl.lit(None, dtype=pl.Int64))
        .otherwise(pl.when(m == 1).then((y - 1) * 100 + 12).otherwise(y * 100 + (m - 1)))
    )

def prev_year_same_month_code_expr(col: str) -> pl.Expr:
    y = pl.col(col) // 100
    m = pl.col(col) % 100
    return (
        pl.when(pl.col(col).is_null())
        .then(pl.lit(None, dtype=pl.Int64))   # CORREÇÃO: usar pl.lit(None, dtype=...)
        .otherwise((y - 1) * 100 + m)
    )

def get_last_n_months_expr(col: str, n: int) -> pl.Expr:
    """
    Retorna uma lista com os códigos das últimas N competências (meses).
    """
    result = []
    for i in range(1, n + 1):
        y = pl.col(col) // 100
        m = pl.col(col) % 100
        
        # Calcular mês i meses atrás
        new_m = m - i
        new_y = y
        
        # Ajustar ano se necessário
        new_m_adjusted = pl.when(new_m <= 0).then(new_m + 12).otherwise(new_m)
        new_y_adjusted = pl.when(new_m <= 0).then(new_y - ((-new_m // 12) + 1)).otherwise(new_y)
        
        comp = new_y_adjusted * 100 + new_m_adjusted
        result.append(comp)
    
    return pl.concat_list(result).alias(f"last_{n}_months")

# exibição/formatos
def brl_format(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return ""

def fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        return f"{x*100:.2f}%".replace(".", ",")
    except Exception:
        return ""

def fmt_comp_str(x):
    if x is None:
        return ""
    try:
        return f"{int(str(x)[:6]):06d}"
    except Exception:
        return str(x)

def format_codigo_str(x):
    if x is None:
        return ""
    try:
        return f"{int(str(x).replace(',', '')):,}".replace(",", ".")  # CORREÇÃO: fechar formato {:,}
    except Exception:
        return str(x).replace(",", ".")

def fmt_int_br(n):
    """Formata inteiros com milhar usando ponto (ex.: 10099 -> '10.099')."""
    try:
        return f"{int(n):,}".replace(",", ".")  # CORREÇÃO: fechar formato {:,}
    except Exception:
        return ""

# ============================
# Fonte via URL (CSV) + Cache
# ============================
st.sidebar.markdown("**Fonte dos dados**")
source_url = st.sidebar.text_input(
    "URL do arquivo CSV (SharePoint/OneDrive)",
    value="https://tcerj365-my.sharepoint.com/:x:/g/personal/emanuellipc_tcerj_tc_br/EfP4k4gSRb5EhjjrRkYyCuwBRy855XAz46Rc6aPgPM1LlA?e=X0wHDt",
    help="Cole o link público do CSV. Forçaremos download direto (download=1)."
)

@st.cache_data(show_spinner=False)
def fetch_bytes(url: str) -> bytes:
    if not url:
        raise ValueError("URL vazia")
    if "download=1" not in url:
        url = url + ("&download=1" if "?" in url else "?download=1")
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
    r = requests.get(url, headers=headers, allow_redirects=True, timeout=180)
    r.raise_for_status()
    return r.content

@st.cache_data(show_spinner=False)
def sniff_delimiter(data: bytes) -> str:
    try:
        sample = data[:20000].decode("utf-8", errors="ignore")
    except Exception:
        sample = ""
    return ";" if sample.count(";") > sample.count(",") else ","

@st.cache_data(show_spinner=False)
def load_csv_polars(data: bytes, sep: str) -> pl.DataFrame:
    return pl.read_csv(
        BytesIO(data),
        separator=sep,
        infer_schema_length=200000,  # melhora inferência em CSVs grandes
        ignore_errors=True,
        try_parse_dates=False,
        low_memory=True
    )

if not source_url:
    st.error("Informe a URL do CSV para continuar.")
    st.stop()

# ============================
# CARREGAMENTO (DEVE VIR ANTES!)
# ============================
with st.spinner("Baixando e carregando a base (cache ativo)..."):
    try:
        raw_bytes = fetch_bytes(source_url)
        sep = sniff_delimiter(raw_bytes)
        df = load_csv_polars(raw_bytes, sep)   # POLARS
    except Exception as e:
        st.error(f"Falha ao carregar a URL: {e}")
        st.stop()

# colunas obrigatórias
required = {"competencia", "descricao", "valor_liquidado"}
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Faltam colunas obrigatórias: {missing}")
    st.stop()

# ============================
# Pré-processamento (Polars)
# ============================
df = df.with_columns([
    expr_to_float_br("valor_liquidado").alias("valor_liquidado"),
    expr_fmt_comp("competencia").alias("competencia_atual"),
])

# ============================
# Configuração (sidebar)
# ============================
with st.sidebar:
    # Botão de logout no topo da sidebar
    st.markdown(f"""
        <div style='text-align: right; color: #666; font-size: 0.85rem; margin-bottom: 1rem;'>
            👤 Usuário: <strong>AdminCIC</strong>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 Sair", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()
    
    st.markdown("---")
    st.header("Configuração")

    candidate_groups = [c for c in ["municipio", "ug", "descricao", "regiao"] if c in df.columns]
    default_groups = [c for c in ["ug", "descricao"] if c in candidate_groups]
    group_cols = st.multiselect(
        "Agrupar por",
        candidate_groups,
        default=default_groups,
        help=(
            "Os limiares estatísticos (IQR/Z/Quantis) e temporais (MoM/YoY) são calculados por grupo "
            "quando você seleciona colunas aqui. Isso impacta diretamente quais pontos serão marcados como outliers."
        ),
    )

    method_options = [
        "Todos os métodos", "IQR", "Z-Score", "Quantis",
        "MoM (mês vs mês anterior)", "YoY (mesmo mês ano anterior)",
        "Média últimos 3 meses", "Média últimos 6 meses",
    ]
    selected_raw = st.multiselect(
        "Métodos de identificação",
        method_options,
        default=["Todos os métodos"],
        help="Se 'Todos os métodos' estiver presente (ou nada), todos serão aplicados."
    )
    all_methods = [m for m in method_options if m != "Todos os métodos"]
    selected_methods = all_methods if ("Todos os métodos" in selected_raw or len(selected_raw) == 0) else selected_raw

    iqr_k = 1.5
    z_thr = 3.0
    q_low, q_high = 0.025, 0.975
    thr_pct = 0.50  # default agora 50%

    if "IQR" in selected_methods:
        iqr_k = st.number_input("k do IQR (Tukey)", min_value=0.1, max_value=10.0, value=iqr_k, step=0.1)
    if "Z-Score" in selected_methods:
        z_thr = st.number_input("|Z| limiar", min_value=0.5, max_value=10.0, value=z_thr, step=0.1)
    if "Quantis" in selected_methods:
        q_low = st.number_input("Quantil inferior", min_value=0.0, max_value=0.49, value=q_low, step=0.005, format="%.3f")
        q_high = st.number_input("Quantil superior", min_value=0.51, max_value=1.0, value=q_high, step=0.005, format="%.3f")
        if q_low >= q_high:
            st.warning("O quantil inferior deve ser menor que o superior.")
    
    # Configuração para métodos temporais (MoM, YoY, 3M, 6M)
    temporal_methods = ["MoM (mês vs mês anterior)", "YoY (mesmo mês ano anterior)", 
                       "Média últimos 3 meses", "Média últimos 6 meses"]
    if any(m in selected_methods for m in temporal_methods):
        thr_pct_choice = st.selectbox(
            "Variação mínima (para métodos temporais)",
            ["25%", "50%", "75%", "100%"],
            index=1,
            help=(
                "Limiar de variação percentual para acionar MoM/YoY/Médias 3M/6M. "
                "Usado em conjunto com a direção dos filtros percentuais abaixo."
            ),
        )
        thr_map = {"25%": 0.25, "50%": 0.50, "75%": 0.75, "100%": 1.00}
        thr_pct = thr_map[thr_pct_choice]

    only_outliers = st.checkbox("Mostrar apenas outliers", value=False)
    st.caption(
        "Um valor é **outlier** se (**MoM** *ou* **YoY** *ou* **3M** *ou* **6M**) **e** (**IQR** *ou* **Z-Score** *ou* **Quantis**) estiverem verdadeiros."
    )

# ============================
# Chaves/joins & métricas (Polars)
# ============================
base = df.with_columns([
    prev_month_code_expr("competencia_atual").alias("ultcomp"),
    prev_year_same_month_code_expr("competencia_atual").alias("comp_ano_anterior"),
])

key_cols = (group_cols + ["competencia_atual"]) if group_cols else ["competencia_atual"]

fact_agg = base.group_by(key_cols, maintain_order=True).agg(
    pl.col("valor_liquidado").sum().alias("valor_liquidado")
)

dim_mom = fact_agg.rename({"competencia_atual": "ultcomp", "valor_liquidado": "valor_ultcomp"})
dim_yoy = fact_agg.rename({"competencia_atual": "comp_ano_anterior", "valor_liquidado": "valor_comp_ano_anterior"})

left_mom_keys = (group_cols + ["ultcomp"]) if group_cols else ["ultcomp"]
left_yoy_keys = (group_cols + ["comp_ano_anterior"]) if group_cols else ["comp_ano_anterior"]

merged = base.join(dim_mom, on=left_mom_keys, how="left").join(dim_yoy, on=left_yoy_keys, how="left")

merged = merged.with_columns([
    pl.when(pl.col("valor_ultcomp").is_not_null() & (pl.col("valor_ultcomp") != 0))
      .then((pl.col("valor_liquidado") - pl.col("valor_ultcomp")) / pl.col("valor_ultcomp"))
      .otherwise(None).alias("pct_change_mom"),
    pl.when(pl.col("valor_comp_ano_anterior").is_not_null() & (pl.col("valor_comp_ano_anterior") != 0))
      .then((pl.col("valor_liquidado") - pl.col("valor_comp_ano_anterior")) / pl.col("valor_comp_ano_anterior"))
      .otherwise(None).alias("pct_change_yoy"),
    pl.when(pl.col("valor_ultcomp").is_not_null())
      .then(pl.col("valor_liquidado") - pl.col("valor_ultcomp"))
      .otherwise(None).alias("dif_mom_reais"),
    pl.when(pl.col("valor_comp_ano_anterior").is_not_null())
      .then(pl.col("valor_liquidado") - pl.col("valor_comp_ano_anterior"))
      .otherwise(None).alias("dif_yoy_reais"),
])

# ============================
# Cálculo médias 3M e 6M
# ============================
if ("Média últimos 3 meses" in selected_methods) or ("Média últimos 6 meses" in selected_methods):
    
    for n_months, method_name in [(3, "Média últimos 3 meses"), (6, "Média últimos 6 meses")]:
        if method_name not in selected_methods:
            continue
        
        # Criar colunas com as competências dos últimos N meses
        temp_merged = merged
        for i in range(1, n_months + 1):
            comp_col = f"comp_back_{i}"
            temp_merged = temp_merged.with_columns([
                prev_month_code_expr("competencia_atual").alias("temp")
            ])
            # Aplicar recursivamente para ir N meses atrás
            for _ in range(i - 1):
                temp_merged = temp_merged.with_columns([
                    prev_month_code_expr("temp").alias("temp")
                ])
            temp_merged = temp_merged.rename({"temp": comp_col})
        
        # Preparar lista de competências para lookup
        comp_cols = [f"comp_back_{i}" for i in range(1, n_months + 1)]
        
        # Criar uma expressão que soma os valores disponíveis dos últimos N meses
        sum_expr = pl.lit(0.0)
        count_expr = pl.lit(0)
        
        for comp_col in comp_cols:
            # Fazer join com fact_agg para cada mês histórico
            lookup_keys = (group_cols + [comp_col]) if group_cols else [comp_col]
            temp_fact = fact_agg.rename({"competencia_atual": comp_col, "valor_liquidado": f"val_{comp_col}"})
            
            temp_merged = temp_merged.join(temp_fact, on=lookup_keys, how="left")
            
            # Somar valores disponíveis
            sum_expr = sum_expr + pl.col(f"val_{comp_col}").fill_null(0.0)
            count_expr = count_expr + pl.when(pl.col(f"val_{comp_col}").is_not_null()).then(1).otherwise(0)
        
        # Calcular média
        avg_col = f"avg_{n_months}m"
        temp_merged = temp_merged.with_columns([
            pl.when(count_expr > 0).then(sum_expr / count_expr).otherwise(None).alias(avg_col)
        ])
        
        # Calcular variações
        pct_col = f"pct_change_{n_months}m"
        dif_col = f"dif_{n_months}m_reais"
        
        temp_merged = temp_merged.with_columns([
            pl.when(pl.col(avg_col).is_not_null() & (pl.col(avg_col) != 0))
              .then((pl.col("valor_liquidado") - pl.col(avg_col)) / pl.col(avg_col))
              .otherwise(None).alias(pct_col),
            pl.when(pl.col(avg_col).is_not_null())
              .then(pl.col("valor_liquidado") - pl.col(avg_col))
              .otherwise(None).alias(dif_col),
        ])
        
        # Limpar colunas temporárias
        cols_to_drop = comp_cols + [f"val_{c}" for c in comp_cols]
        temp_merged = temp_merged.drop([c for c in cols_to_drop if c in temp_merged.columns])
        
        # Atualizar merged
        merged = temp_merged

# ============================
# Outliers (Polars vetorizado)
# ============================
for col in ["out_iqr", "out_z", "out_q", "out_mom", "out_yoy", "out_3m", "out_6m"]:
    merged = merged.with_columns(pl.lit(False).alias(col))

group_over = group_cols if group_cols else None

if "IQR" in selected_methods:
    q1 = pl.col("valor_liquidado").quantile(0.25).over(group_over) if group_over else pl.col("valor_liquidado").quantile(0.25)
    q3 = pl.col("valor_liquidado").quantile(0.75).over(group_over) if group_over else pl.col("valor_liquidado").quantile(0.75)
    iqr = (q3 - q1)
    low = q1 - iqr_k * iqr
    high = q3 + iqr_k * iqr
    merged = merged.with_columns(((pl.col("valor_liquidado") < low) | (pl.col("valor_liquidado") > high)).alias("out_iqr"))

if "Z-Score" in selected_methods:
    mu = pl.col("valor_liquidado").mean().over(group_over) if group_over else pl.col("valor_liquidado").mean()
    sd = pl.col("valor_liquidado").std(ddof=0).over(group_over) if group_over else pl.col("valor_liquidado").std(ddof=0)
    z = (pl.col("valor_liquidado") - mu) / sd
    merged = merged.with_columns(pl.when(sd.is_null() | (sd == 0)).then(False).otherwise(z.abs() > z_thr).alias("out_z"))

if "Quantis" in selected_methods:
    lo = pl.col("valor_liquidado").quantile(q_low).over(group_over) if group_over else pl.col("valor_liquidado").quantile(q_low)
    hi = pl.col("valor_liquidado").quantile(q_high).over(group_over) if group_over else pl.col("valor_liquidado").quantile(q_high)
    merged = merged.with_columns(((pl.col("valor_liquidado") < lo) | (pl.col("valor_liquidado") > hi)).alias("out_q"))

if "MoM (mês vs mês anterior)" in selected_methods:
    merged = merged.with_columns((pl.col("pct_change_mom").abs() >= thr_pct).fill_null(False).alias("out_mom"))

if "YoY (mesmo mês ano anterior)" in selected_methods:
    merged = merged.with_columns((pl.col("pct_change_yoy").abs() >= thr_pct).fill_null(False).alias("out_yoy"))

if "Média últimos 3 meses" in selected_methods:
    merged = merged.with_columns((pl.col("pct_change_3m").abs() >= thr_pct).fill_null(False).alias("out_3m"))

if "Média últimos 6 meses" in selected_methods:
    merged = merged.with_columns((pl.col("pct_change_6m").abs() >= thr_pct).fill_null(False).alias("out_6m"))

# Regra final: (temporal) AND (estatístico)
merged = merged.with_columns([
    (pl.col("out_mom") | pl.col("out_yoy") | pl.col("out_3m") | pl.col("out_6m")).alias("flag_temporal"),
    (pl.col("out_iqr") | pl.col("out_z") | pl.col("out_q")).alias("flag_estatistico"),
])
merged = merged.with_columns(
    (pl.col("flag_temporal") & pl.col("flag_estatistico")).alias("outlier")
)

# Adicionar coluna de comentário explicativo
merged = merged.with_columns([
    pl.when(pl.col("outlier") == True)
      .then(pl.lit("✓ Outlier: Atende critérios temporal E estatístico"))
    .when(pl.col("flag_temporal") == True)
      .then(
          pl.when(pl.col("flag_estatistico") == False)
            .then(pl.lit("⚠ Mudança temporal detectada, mas valor dentro da distribuição normal do grupo"))
            .otherwise(pl.lit(""))
      )
    .when(pl.col("flag_estatistico") == True)
      .then(
          pl.when(pl.col("flag_temporal") == False)
            .then(pl.lit("⚠ Valor estatisticamente extremo, mas sem mudança temporal significativa"))
            .otherwise(pl.lit(""))
      )
    .otherwise(pl.lit(""))
    .alias("diagnostico")
])

# ============================
# Filtros (Polars)
# ============================
with st.sidebar:
    st.header("Filtros")
    filtered = merged

    def sorted_unique(series: pl.Series):
        vals = [x for x in series.drop_nulls().unique().to_list() if str(x).strip() != ""]
        try:
            return sorted(vals)
        except Exception:
            return vals

    if "municipio" in filtered.columns:
        mun_opts = sorted_unique(filtered["municipio"])
        sel_mun = st.multiselect("Município", mun_opts, default=[])
        if sel_mun:
            filtered = filtered.filter(pl.col("municipio").is_in(sel_mun))

    if "ug" in filtered.columns:
        ug_opts = sorted_unique(filtered["ug"])
        sel_ug = st.multiselect("Unidade Gestora", ug_opts, default=[])
        if sel_ug:
            filtered = filtered.filter(pl.col("ug").is_in(sel_ug))

    if "descricao" in filtered.columns:
        desc_opts = sorted_unique(filtered["descricao"])
        sel_desc = st.multiselect("Descrição da Despesa", desc_opts, default=[])
        if sel_desc:
            filtered = filtered.filter(pl.col("descricao").is_in(sel_desc))

    # Competência (AAAAMM)
    comp_opts = filtered["competencia_atual"].drop_nulls().cast(pl.Int64, strict=False).to_list()
    comp_opts = sorted({fmt_comp_str(c) for c in comp_opts})
    sel_comp = st.multiselect("Competência (AAAAMM)", comp_opts, default=[])
    if sel_comp:
        comp_ints = [int(c) for c in sel_comp]
        filtered = filtered.filter(pl.col("competencia_atual").is_in(comp_ints))

    # Ano
    filtered = filtered.with_columns(((pl.col("competencia_atual") // 100).cast(pl.Int64)).alias("ano_num"))
    ano_opts = sorted_unique(filtered["ano_num"])
    # Default dinâmico: dois anos mais recentes
    default_years = sorted(ano_opts)[-2:] if len(ano_opts) > 0 else []
    sel_ano = st.multiselect("Ano (AAAA)", ano_opts, default=default_years)
    if sel_ano:
        filtered = filtered.filter(pl.col("ano_num").is_in(sel_ano))

    # ===== NOVO: Faixa Valor Liquidado (inclui 'Todos' e corrige 'Até 40 mil' para 0..40k) =====
    faixa_val_liq_options = [
        "Todos",
        "Valores Negativos",
        "Até R$ 40 mil",
        "Acima de R$ 40 mil",
    ]

    def apply_value_range_filter_val_liq(df_pl: pl.DataFrame, column: str, choice: str) -> pl.DataFrame:
        if choice == "Valores Negativos":
            return df_pl.filter(pl.col(column) < 0)
        elif choice == "Até R$ 40 mil":
            return df_pl.filter((pl.col(column) >= 0) & (pl.col(column) <= 40_000))
        elif choice == "Acima de R$ 40 mil":
            return df_pl.filter(pl.col(column) > 40_000)
        return df_pl

    val_choice = st.selectbox(
        "Faixa Valor Liquidado (R$)",
        options=faixa_val_liq_options,
        index=0,
        help="Use 'Todos' para não filtrar. A opção 'Até R$ 40 mil' considera apenas valores não-negativos.",
    )
    if val_choice != "Todos":
        filtered = apply_value_range_filter_val_liq(filtered, "valor_liquidado", val_choice)

    # ===== Modo por magnitude e direção para Diferenças em R$ =====
    use_abs_diff = st.checkbox("Filtrar Diferenças por magnitude (usar |dif|)", value=False)
    dir_rs = st.selectbox(
        "Direção das diferenças (R$)",
        ["Todos (sinal indiferente)", "Aumentos (> 0)", "Quedas (< 0)"],
        index=0,
        help=(
            "Quando 'por magnitude' estiver ativo, as faixas em R$ usam |dif|. "
            "Você pode ainda restringir por direção positiva/negativa."
        ),
    )

    def apply_direction(df_pl: pl.DataFrame, column: str, direction: str) -> pl.DataFrame:
        if direction == "Aumentos (> 0)":
            return df_pl.filter(pl.col(column) > 0)
        if direction == "Quedas (< 0)":
            return df_pl.filter(pl.col(column) < 0)
        return df_pl

    # Faixas R$ (labels padronizadas e menos restritivas)
    faixa_diff_options = [
        "Todos",
        "Valores Negativos",
        "Até R$ 40 mil",
        "Acima de R$ 250 mil",
        "Acima de R$ 500 mil",
        "Acima de R$ 750 mil",
    ]

    def apply_value_range_filter(df_pl: pl.DataFrame, column: str, choice: str, use_abs: bool) -> pl.DataFrame:
        col = pl.col(column).abs() if use_abs else pl.col(column)
        if choice == "Valores Negativos" and not use_abs:
            return df_pl.filter(pl.col(column) < 0)
        elif choice == "Até R$ 40 mil":
            return df_pl.filter((col >= 0) & (col <= 40_000))
        elif choice == "Acima de R$ 250 mil":
            return df_pl.filter(col > 250_000)
        elif choice == "Acima de R$ 500 mil":
            return df_pl.filter(col > 500_000)
        elif choice == "Acima de R$ 750 mil":
            return df_pl.filter(col > 750_000)
        return df_pl

    include_nulls = st.checkbox(
        "Incluir linhas sem base de comparação (nulos)",
        value=True,
        help="Quando desmarcado, linhas com diferenças nulas (sem mês anterior/ano anterior) serão removidas ao aplicar filtros de Dif.",
    )

    dm_choice = st.selectbox("Faixa Dif. MoM (R$)", options=faixa_diff_options, index=0)
    if dm_choice != "Todos":
        non_null_dm = filtered.filter(pl.col("dif_mom_reais").is_not_null())
        non_null_dm = apply_direction(non_null_dm, "dif_mom_reais", dir_rs)
        filtered_dm = apply_value_range_filter(non_null_dm, "dif_mom_reais", dm_choice, use_abs_diff)
        if include_nulls:
            null_dm = filtered.filter(pl.col("dif_mom_reais").is_null())
            filtered = pl.concat([filtered_dm, null_dm], how="vertical")
        else:
            filtered = filtered_dm

    dy_choice = st.selectbox("Faixa Dif. YoY (R$)", options=faixa_diff_options, index=0)
    if dy_choice != "Todos":
        non_null_dy = filtered.filter(pl.col("dif_yoy_reais").is_not_null())
        non_null_dy = apply_direction(non_null_dy, "dif_yoy_reais", dir_rs)
        filtered_dy = apply_value_range_filter(non_null_dy, "dif_yoy_reais", dy_choice, use_abs_diff)
        if include_nulls:
            null_dy = filtered.filter(pl.col("dif_yoy_reais").is_null())
            filtered = pl.concat([filtered_dy, null_dy], how="vertical")
        else:
            filtered = filtered_dy

    # ===== Percentuais com direção =====
    pct_options = ["Todos", "Até 30%", "30 a 70%", "70 a 100%", "Acima de 100%"]

    dir_pct = st.selectbox(
        "Direção das variações (%)",
        ["Ambos (abs)", "Aumentos (var > 0)", "Quedas (var < 0)"],
        index=0,
        help=(
            "Filtros de % usam valor absoluto por padrão (Ambos). "
            "Escolha 'Aumentos' para variações positivas ou 'Quedas' para negativas."
        ),
    )

    def apply_pct_range_filter(df_pl: pl.DataFrame, column: str, choice: str, direction: str) -> pl.DataFrame:
        col = pl.col(column).abs() if direction == "Ambos (abs)" else pl.col(column)
        if choice == "Até 30%":
            return df_pl.filter(col <= 0.30)
        elif choice == "30 a 70%":
            return df_pl.filter((col > 0.30) & (col <= 0.70))
        elif choice == "70 a 100%":
            return df_pl.filter((col > 0.70) & (col <= 1.0))
        elif choice == "Acima de 100%":
            return df_pl.filter(col > 1.0)
        return df_pl

    def apply_pct_direction(df_pl: pl.DataFrame, column: str, direction: str) -> pl.DataFrame:
        if direction == "Aumentos (var > 0)":
            return df_pl.filter(pl.col(column) > 0)
        if direction == "Quedas (var < 0)":
            return df_pl.filter(pl.col(column) < 0)
        return df_pl

    pct_mom_choice = st.selectbox("Faixa Dif. MoM (%)", options=pct_options, index=0)
    if pct_mom_choice != "Todos":
        non_null_pct_mom = filtered.filter(pl.col("pct_change_mom").is_not_null())
        non_null_pct_mom = apply_pct_direction(non_null_pct_mom, "pct_change_mom", dir_pct)
        filtered_pct_mom = apply_pct_range_filter(non_null_pct_mom, "pct_change_mom", pct_mom_choice, dir_pct)
        if include_nulls:
            null_pct_mom = filtered.filter(pl.col("pct_change_mom").is_null())
            filtered = pl.concat([filtered_pct_mom, null_pct_mom], how="vertical")
        else:
            filtered = filtered_pct_mom

    pct_yoy_choice = st.selectbox("Faixa Dif. YoY (%)", options=pct_options, index=0)
    if pct_yoy_choice != "Todos":
        non_null_pct_yoy = filtered.filter(pl.col("pct_change_yoy").is_not_null())
        non_null_pct_yoy = apply_pct_direction(non_null_pct_yoy, "pct_change_yoy", dir_pct)
        filtered_pct_yoy = apply_pct_range_filter(non_null_pct_yoy, "pct_change_yoy", pct_yoy_choice, dir_pct)
        if include_nulls:
            null_pct_yoy = filtered.filter(pl.col("pct_change_yoy").is_null())
            filtered = pl.concat([filtered_pct_yoy, null_pct_yoy], how="vertical")
        else:
            filtered = filtered_pct_yoy

    # Filtros para Média 3 Meses (se disponível)
    if "dif_3m_reais" in filtered.columns:
        st.markdown("**Filtros Média 3 Meses**")
        dm_3m_choice = st.selectbox("Faixa Dif. Média 3M (R$)", options=faixa_diff_options, index=0, key="dm_3m")
        if dm_3m_choice != "Todos":
            non_null_dm3 = filtered.filter(pl.col("dif_3m_reais").is_not_null())
            non_null_dm3 = apply_direction(non_null_dm3, "dif_3m_reais", dir_rs)
            filtered_dm3 = apply_value_range_filter(non_null_dm3, "dif_3m_reais", dm_3m_choice, use_abs_diff)
            if include_nulls:
                null_dm3 = filtered.filter(pl.col("dif_3m_reais").is_null())
                filtered = pl.concat([filtered_dm3, null_dm3], how="vertical")
            else:
                filtered = filtered_dm3

        pct_3m_choice = st.selectbox("Faixa Dif. Média 3M (%)", options=pct_options, index=0, key="pct_3m")
        if pct_3m_choice != "Todos":
            non_null_pct3 = filtered.filter(pl.col("pct_change_3m").is_not_null())
            non_null_pct3 = apply_pct_direction(non_null_pct3, "pct_change_3m", dir_pct)
            filtered_pct3 = apply_pct_range_filter(non_null_pct3, "pct_change_3m", pct_3m_choice, dir_pct)
            if include_nulls:
                null_pct3 = filtered.filter(pl.col("pct_change_3m").is_null())
                filtered = pl.concat([filtered_pct3, null_pct3], how="vertical")
            else:
                filtered = filtered_pct3

    # Filtros para Média 6 Meses (se disponível)
    if "dif_6m_reais" in filtered.columns:
        st.markdown("**Filtros Média 6 Meses**")
        dm_6m_choice = st.selectbox("Faixa Dif. Média 6M (R$)", options=faixa_diff_options, index=0, key="dm_6m")
        if dm_6m_choice != "Todos":
            non_null_dm6 = filtered.filter(pl.col("dif_6m_reais").is_not_null())
            non_null_dm6 = apply_direction(non_null_dm6, "dif_6m_reais", dir_rs)
            filtered_dm6 = apply_value_range_filter(non_null_dm6, "dif_6m_reais", dm_6m_choice, use_abs_diff)
            if include_nulls:
                null_dm6 = filtered.filter(pl.col("dif_6m_reais").is_null())
                filtered = pl.concat([filtered_dm6, null_dm6], how="vertical")
            else:
                filtered = filtered_dm6

        pct_6m_choice = st.selectbox("Faixa Dif. Média 6M (%)", options=pct_options, index=0, key="pct_6m")
        if pct_6m_choice != "Todos":
            non_null_pct6 = filtered.filter(pl.col("pct_change_6m").is_not_null())
            non_null_pct6 = apply_pct_direction(non_null_pct6, "pct_change_6m", dir_pct)
            filtered_pct6 = apply_pct_range_filter(non_null_pct6, "pct_change_6m", pct_6m_choice, dir_pct)
            if include_nulls:
                null_pct6 = filtered.filter(pl.col("pct_change_6m").is_null())
                filtered = pl.concat([filtered_pct6, null_pct6], how="vertical")
            else:
                filtered = filtered_pct6

# ============================
# Display formatado
# ============================
out = filtered

if only_outliers:
    out = out.filter(pl.col("outlier") == True)

if "competencia" in out.columns:
    out = out.drop("competencia")

out = out.with_columns([
    pl.when(pl.col("competencia_atual").is_not_null())
      .then(pl.col("competencia_atual").cast(pl.Int64).cast(pl.Utf8).str.slice(0, 6))
      .otherwise(pl.lit("")).alias("competencia_atual_str"),
    pl.when(pl.col("ultcomp").is_not_null())
      .then(pl.col("ultcomp").cast(pl.Int64).cast(pl.Utf8).str.slice(0, 6))
      .otherwise(pl.lit("")).alias("ultcomp_str"),
    pl.when(pl.col("comp_ano_anterior").is_not_null())
      .then(pl.col("comp_ano_anterior").cast(pl.Int64).cast(pl.Utf8).str.slice(0, 6))
      .otherwise(pl.lit("")).alias("comp_ano_anterior_str"),
])

out = out.with_columns([
    pl.when(pl.col("competencia_atual_str").str.len_chars() >= 4)
      .then(pl.col("competencia_atual_str").str.slice(0, 4))
      .otherwise(pl.lit("")).alias("ano")
])

for col in ["valor_liquidado", "valor_ultcomp", "valor_comp_ano_anterior", "dif_mom_reais", "dif_yoy_reais", "avg_3m", "avg_6m", "dif_3m_reais", "dif_6m_reais"]:
    if col in out.columns:
        out = out.with_columns(pl.col(col).map_elements(brl_format, return_dtype=pl.Utf8).alias(col))
for col in ["pct_change_mom", "pct_change_yoy", "pct_change_3m", "pct_change_6m"]:
    if col in out.columns:
        out = out.with_columns(pl.col(col).map_elements(fmt_pct, return_dtype=pl.Utf8).alias(col))
if "codigo" in out.columns:
    out = out.with_columns(pl.col("codigo").map_elements(format_codigo_str, return_dtype=pl.Utf8).alias("codigo"))

final_order = [
    "municipio", "ug", "ano", "regiao",
    "codigo", "descricao", "ultcomp_str", "comp_ano_anterior_str",
    "competencia_atual_str",  # movida para aparecer APÓS 'Competência Ano Anterior' e ANTES de 'Valor Liquidado'
    "valor_liquidado", "valor_ultcomp", "valor_comp_ano_anterior",
    "avg_3m", "avg_6m",
    "dif_mom_reais", "dif_yoy_reais", "dif_3m_reais", "dif_6m_reais",
    "pct_change_mom", "pct_change_yoy", "pct_change_3m", "pct_change_6m",
    "outlier", "diagnostico", "out_iqr", "out_z", "out_q", "out_mom", "out_yoy", "out_3m", "out_6m"
]

present_cols = [c for c in final_order if c in out.columns]
rename_map = {
    "competencia_atual_str": "Competência Atual",
    "municipio": "Município",
    "ug": "Unidade Gestora",
    "ano": "Ano",
    "regiao": "Região Administrativa",
    "codigo": "Código",
    "descricao": "Descrição da Despesa",
    "ultcomp_str": "Última Competência",
    "comp_ano_anterior_str": "Competência Ano Anterior",
    "valor_liquidado": "Valor Liquidado",
    "valor_ultcomp": "Valor da Última Competência",
    "valor_comp_ano_anterior": "Valor do Ano Anterior",
    "avg_3m": "Média Últimos 3 Meses",
    "avg_6m": "Média Últimos 6 Meses",
    "dif_mom_reais": "Diferença Mensal em Reais",
    "dif_yoy_reais": "Diferença Anual em Reais",
    "dif_3m_reais": "Diferença vs Média 3M em Reais",
    "dif_6m_reais": "Diferença vs Média 6M em Reais",
    "pct_change_mom": "Diferença Mensal Percentual",
    "pct_change_yoy": "Diferença Anual Percentual",
    "pct_change_3m": "Diferença vs Média 3M Percentual",
    "pct_change_6m": "Diferença vs Média 6M Percentual",
    "outlier": "Outlier",
    "diagnostico": "Diagnóstico",
}

out_to_show = out.select(present_cols).rename(rename_map).to_pandas()

st.success("Base carregada com Sucesso!")
st.dataframe(out_to_show, use_container_width=True)

# ============================
# Métricas
# ============================

total_registros = filtered.height
total_outliers = int(filtered["outlier"].sum())
pct_outliers = (total_outliers / total_registros) if total_registros > 0 else 0.0

def metric_card(label: str, value: str) -> str:
    return f"""
    <div style='border:1px solid #ddd; border-radius:6px; padding:12px; background-color:#f9f9f9; text-align:center;'>
        <div style='font-size:0.9rem; color:#666;'>{label}</div>
        <div style='font-size:1.6rem; font-weight:bold; color:#333;'>{value}</div>
    </div>
    """

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(metric_card("Total de Registros", f"{total_registros}"), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card("Total de Outliers", f"{total_outliers}"), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card("% Outliers", f"{pct_outliers:.1%}"), unsafe_allow_html=True)

# ============================
# Download
# ============================
@st.cache_data
def to_csv_bytes(df_pd) -> bytes:
    return df_pd.to_csv(index=False).encode("utf-8")

st.download_button(
    "Baixar (CSV filtrado)",
    to_csv_bytes(out_to_show),
    file_name="competencias_outliers_hibrido.csv",
    mime="text/csv"
)

# ============================
# Abas adicionais
# ============================
try:
    import altair as alt
except Exception:
    alt = None

abas = st.tabs(["Análise Gráfica", "Análise Avançada", "Sobre os Métodos"])
aba, aba_avancada, aba2 = abas[0], abas[1], abas[2]

with aba:
    st.subheader("Outliers por critério (combinação de métodos)")

    # Coluna com a COMBINAÇÃO exata de métodos acionados
    comb = (
        pl.concat_str(
            [
                pl.when(pl.col("out_yoy")).then(pl.lit("YoY")).otherwise(pl.lit("")),
                pl.when(pl.col("out_mom")).then(pl.lit("MoM")).otherwise(pl.lit("")),
                pl.when(pl.col("out_3m") if "out_3m" in filtered.columns else pl.lit(False)).then(pl.lit("3M")).otherwise(pl.lit("")),
                pl.when(pl.col("out_6m") if "out_6m" in filtered.columns else pl.lit(False)).then(pl.lit("6M")).otherwise(pl.lit("")),
                pl.when(pl.col("out_iqr")).then(pl.lit("IQR")).otherwise(pl.lit("")),
                pl.when(pl.col("out_q")).then(pl.lit("Quantis")).otherwise(pl.lit("")),
                pl.when(pl.col("out_z")).then(pl.lit("Z-Score")).otherwise(pl.lit("")),
            ],
            separator=" + ",
        )
        .str.replace_all(r"(\s*\+\s*)+", " + ")
        .str.replace_all(r"^\s*\+\s*|\s*\+\s*$", "")
        .str.replace_all(r"\s+", " ")
    ).alias("Critério")

    comb_counts = (
        filtered
        .with_columns([comb])
        .filter(pl.col("outlier") == True)
        .group_by("Critério")
        .agg(pl.count().alias("Outliers"))
        .sort("Outliers", descending=True)
    )

    if comb_counts.height == 0:
        st.info("Nenhum outlier encontrado com as combinações atuais.")
    else:
        cc_df = comb_counts.to_pandas()
        st.dataframe(cc_df, use_container_width=True)

    # ============================
    # GRÁFICOS EMPILHADOS + TABELAS (100% Polars até Altair)
    # ============================
    def has_col(df_pl: pl.DataFrame, col: str) -> bool:
        return col in df_pl.columns

    def build_stacked_section(title: str, df_pl: pl.DataFrame, group_col: str, top_n: int | None = None):
        if not has_col(df_pl, group_col):
            st.info(f"Coluna '{group_col}' não encontrada na base.")
            return

        base = (
            df_pl
            .select([
                pl.col(group_col).cast(pl.Utf8).fill_null('').alias(group_col),
                pl.col('valor_liquidado').fill_null(0.0).alias('valor_liquidado'),
                pl.col('outlier').fill_null(False).alias('outlier'),
            ])
            .with_columns([
                pl.when(pl.col(group_col).str.to_lowercase() == 'outras')
                  .then(pl.lit('Outras - Consórcios'))
                  .otherwise(pl.col(group_col))
                  .alias(group_col)
            ])
        )

        totais = (
            base.group_by(group_col)
                .agg(pl.col('valor_liquidado').sum().alias('total'))
                .sort('total', descending=True)
        )

        if top_n is not None:
            top_keys = totais.head(top_n)[group_col]
            base = base.filter(pl.col(group_col).is_in(top_keys))
            totais = totais.filter(pl.col(group_col).is_in(top_keys))

        stacked_val = (
            base.group_by([group_col, 'outlier'])
                .agg(pl.col('valor_liquidado').sum().alias('valor'))
        )

        total_cnt_by_group = base.group_by(group_col).agg(pl.count().alias('qtde_total'))
        out_cnt_by_group = base.filter(pl.col('outlier') == True).group_by(group_col).agg(pl.count().alias('qtde_outliers'))

        cnts = (
            total_cnt_by_group
            .join(out_cnt_by_group, on=group_col, how='left')
            .with_columns([
                pl.col('qtde_outliers').fill_null(0).alias('qtde_outliers'),
                (pl.col('qtde_total') - pl.col('qtde_outliers')).alias('qtde_nao_outliers'),
                pl.when(pl.col('qtde_total') > 0)
                  .then(pl.col('qtde_outliers') / pl.col('qtde_total'))
                  .otherwise(0.0).alias('pct_outliers')
            ])
        )

        grupos = totais.select(group_col)
        cat_true = grupos.with_columns(pl.lit(True).alias('outlier'))
        cat_false = grupos.with_columns(pl.lit(False).alias('outlier'))
        full = pl.concat([cat_true, cat_false], how='vertical')

        stacked = (
            full.join(stacked_val, on=[group_col, 'outlier'], how='left')
                .with_columns(pl.col('valor').fill_null(0.0))
                .join(cnts, on=group_col, how='left')
                .with_columns([
                    pl.col('qtde_total').fill_null(0),
                    pl.col('qtde_outliers').fill_null(0),
                    pl.col('qtde_nao_outliers').fill_null(0),
                    pl.col('pct_outliers').fill_null(0.0),
                ])
        )

        stacked = (
            stacked.join(totais, on=group_col, how='left')
                   .sort(['total', group_col], descending=[True, False])
                   .with_columns([
                       pl.when(pl.col('outlier') == True).then(pl.lit('Outliers')).otherwise(pl.lit('Não-Outliers')).alias('Categoria'),
                       pl.col(group_col).alias('Grupo'),
                       pl.col('valor').alias('Valor'),
                   ])
                   .select(['Grupo', 'Categoria', 'Valor', 'total'])
        )

        st.markdown(f"### {title}")
        if alt is not None:
            chart_df = stacked.to_pandas()
            # Adicionar coluna formatada para tooltip
            chart_df['Valor_fmt'] = chart_df['Valor'].apply(brl_format)
            
            # Calcular % de outliers por grupo para o tooltip
            outliers_by_group = chart_df[chart_df['Categoria'] == 'Outliers'].groupby('Grupo')['Valor'].sum()
            total_by_group = chart_df.groupby('Grupo')['Valor'].sum()
            pct_by_group = ((outliers_by_group / total_by_group) * 100).fillna(0)
            
            # Adicionar % formatado ao dataframe
            chart_df['Pct_Outliers'] = chart_df['Grupo'].map(lambda g: f"{pct_by_group.get(g, 0):.2f}%".replace('.', ','))
            
            # Criar ordem customizada baseada no total (decrescente)
            order_df = chart_df.groupby('Grupo')['total'].first().sort_values(ascending=False)
            grupo_order = order_df.index.tolist()
            
            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    y=alt.Y('Grupo:N', title='', sort=grupo_order),
                    x=alt.X('Valor:Q', title='Valor Liquidado (R$)', stack='zero'),
                    color=alt.Color('Categoria:N', legend=alt.Legend(title='')),
                    tooltip=[
                        alt.Tooltip('Grupo:N', title='Grupo'),
                        alt.Tooltip('Categoria:N', title='Categoria'),
                        alt.Tooltip('Valor_fmt:N', title='Valor'),
                        alt.Tooltip('Pct_Outliers:N', title='% Outliers')
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning('Altair não disponível. Exibindo apenas tabela abaixo.')

        # Espaçamento entre gráfico e tabela
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Hint sobre ordenação
        st.caption("📊 Tabela ordenada por valor de Outliers (decrescente)")
        
        tab = (
            stacked.group_by('Grupo')
                   .agg([
                       pl.col('Valor').sum().alias('Total'),
                       pl.when(pl.col('Categoria') == 'Outliers').then(pl.col('Valor')).otherwise(0.0).sum().alias('Outliers'),
                       pl.when(pl.col('Categoria') == 'Não-Outliers').then(pl.col('Valor')).otherwise(0.0).sum().alias('Não-Outliers'),
                   ])
                   .with_columns([
                       pl.when(pl.col('Total') > 0)
                         .then((pl.col('Outliers') / pl.col('Total')) * 100)
                         .otherwise(0.0)
                         .alias('% Outliers')
                   ])
                   .sort('Outliers', descending=True)
                   .to_pandas()
        )
        # Guardar valores numéricos para % antes de formatar
        pct_outliers_values = tab['% Outliers'].copy()
        
        tab['Total'] = [brl_format(v) for v in tab['Total']]
        tab['Outliers'] = [brl_format(v) for v in tab['Outliers']]
        tab['Não-Outliers'] = [brl_format(v) for v in tab['Não-Outliers']]
        tab['% Outliers'] = [f"{v:.2f}%".replace('.', ',') for v in pct_outliers_values]
        st.dataframe(tab, use_container_width=True)
        st.markdown("<br><br>", unsafe_allow_html=True)

    if 'regiao' in filtered.columns:
        build_stacked_section('Distribuíção de Outliers x Região Administrativa', filtered, 'regiao', None)
    if 'municipio' in filtered.columns:
        build_stacked_section('Top 10 Municípios', filtered, 'municipio', 10)
    if 'ug' in filtered.columns:
        build_stacked_section('Top 10 Unidades Gestoras', filtered, 'ug', 10)

with aba_avancada:
    st.subheader("Análise Avançada")

    try:
        import altair as alt
    except Exception:
        alt = None

    st.markdown("#### Soma do Valor Liquidado por Competência")
    if alt is not None and "competencia_atual" in filtered.columns:
        serie1 = (
            filtered
            .select([
                pl.col("competencia_atual").alias("comp"),
                pl.col("valor_liquidado").fill_null(0.0).alias("vl")
            ])
            .group_by("comp")
            .agg(pl.col("vl").sum().alias("Valor Liquidado"))
            .sort("comp")
            .with_columns(pl.col("comp").cast(pl.Int64).cast(pl.Utf8).str.slice(0, 6).alias("Competência"))
            .select(["Competência", "Valor Liquidado"])
        )
        df1 = serie1.to_pandas()
        df1['Valor_fmt'] = df1['Valor Liquidado'].apply(brl_format)  # adicionado: coluna formatada
        chart1 = (
            alt.Chart(df1)
            .mark_line(point=True)
            .encode(
                x=alt.X("Competência:N", title="Competência (AAAAMM)"),
                y=alt.Y("Valor Liquidado:Q", title="Soma do Valor Liquidado (R$)"),
                tooltip=[alt.Tooltip("Competência:N", title="Competência"),
                         alt.Tooltip("Valor_fmt:N", title="Valor")],  # usa formato R$
            )
            .properties(height=320)
        )
        st.altair_chart(chart1, use_container_width=True)
    else:
        st.info("Dados de competência não disponíveis para o gráfico de linha.")

    st.markdown("#### Comparativo por Competência: Atual × Última Competência × Ano Anterior")
    needed_cols = {"competencia_atual", "valor_liquidado", "valor_ultcomp", "valor_comp_ano_anterior"}
    if alt is not None and needed_cols.issubset(set(filtered.columns)):
        agg = (
            filtered
            .select([
                pl.col("competencia_atual").alias("comp"),
                pl.col("valor_liquidado").fill_null(0.0).alias("valor_liquidado"),
                pl.col("valor_ultcomp").fill_null(0.0).alias("valor_ultcomp"),
                pl.col("valor_comp_ano_anterior").fill_null(0.0).alias("valor_comp_ano_anterior"),
            ])
            .group_by("comp")
            .agg([
                pl.col("valor_liquidado").sum().alias("Valor Liquidado"),
                pl.col("valor_ultcomp").sum().alias("Valor Última Competência"),
                pl.col("valor_comp_ano_anterior").sum().alias("Valor Ano Anterior"),
            ])
            .sort("comp")
            .with_columns(pl.col("comp").cast(pl.Int64).cast(pl.Utf8).str.slice(0, 6).alias("Competência"))
        )

        long = agg.melt(
            id_vars=["Competência"],
            value_vars=["Valor Liquidado", "Valor Última Competência", "Valor Ano Anterior"],
            variable_name="Série",
            value_name="Valor"
        )

        df2 = long.to_pandas()
        df2['Valor_fmt'] = df2['Valor'].apply(brl_format)  # adicionado: coluna formatada

        chart2 = (
            alt.Chart(df2)
            .mark_line(point=True)
            .encode(
                x=alt.X("Competência:N", title="Competência (AAAAMM)"),
                y=alt.Y("Valor:Q", title="Valor (R$)"),
                color=alt.Color("Série:N", title="Série"),
                tooltip=[alt.Tooltip("Competência:N", title="Competência"),
                         alt.Tooltip("Série:N", title="Série"),
                         alt.Tooltip("Valor_fmt:N", title="Valor")],  # usa formato R$
            )
            .properties(height=340)
        )
        st.altair_chart(chart2, use_container_width=True)
    else:
        st.info("Colunas necessárias não disponíveis para o comparativo.")

    st.markdown("#### Comparativo por Competência: Atual × Média 3M × Média 6M")
    needed_cols_avg = {"competencia_atual", "valor_liquidado"}
    has_3m = "avg_3m" in filtered.columns
    has_6m = "avg_6m" in filtered.columns
    
    if alt is not None and needed_cols_avg.issubset(set(filtered.columns)) and (has_3m or has_6m):
        select_cols = [
            pl.col("competencia_atual").alias("comp"),
            pl.col("valor_liquidado").fill_null(0.0).alias("valor_liquidado"),
        ]
        agg_cols = [pl.col("valor_liquidado").sum().alias("Valor Liquidado Atual")]
        value_vars = ["Valor Liquidado Atual"]
        
        if has_3m:
            select_cols.append(pl.col("avg_3m").fill_null(0.0).alias("avg_3m"))
            agg_cols.append(pl.col("avg_3m").sum().alias("Média Últimos 3 Meses"))
            value_vars.append("Média Últimos 3 Meses")
        
        if has_6m:
            select_cols.append(pl.col("avg_6m").fill_null(0.0).alias("avg_6m"))
            agg_cols.append(pl.col("avg_6m").sum().alias("Média Últimos 6 Meses"))
            value_vars.append("Média Últimos 6 Meses")
        
        agg_avg = (
            filtered
            .select(select_cols)
            .group_by("comp")
            .agg(agg_cols)
            .sort("comp")
            .with_columns(pl.col("comp").cast(pl.Int64).cast(pl.Utf8).str.slice(0, 6).alias("Competência"))
        )

        long_avg = agg_avg.melt(
            id_vars=["Competência"],
            value_vars=value_vars,
            variable_name="Série",
            value_name="Valor"
        )

        df_avg = long_avg.to_pandas()
        df_avg['Valor_fmt'] = df_avg['Valor'].apply(brl_format)

        chart_avg = (
            alt.Chart(df_avg)
            .mark_line(point=True)
            .encode(
                x=alt.X("Competência:N", title="Competência (AAAAMM)"),
                y=alt.Y("Valor:Q", title="Valor (R$)"),
                color=alt.Color("Série:N", title="Série"),
                tooltip=[alt.Tooltip("Competência:N", title="Competência"),
                         alt.Tooltip("Série:N", title="Série"),
                         alt.Tooltip("Valor_fmt:N", title="Valor")],
            )
            .properties(height=340)
        )
        st.altair_chart(chart_avg, use_container_width=True)
    else:
        st.info("Médias 3M e/ou 6M não disponíveis. Ative esses métodos na configuração para visualizar este gráfico.")

with aba2:
    st.subheader("📘 Explicação dos Métodos de Detecção de Outliers")
    st.markdown(
        """
Os métodos abaixo servem para **identificar valores que se desviam fortemente do padrão esperado**. 
Cada método tem uma lógica distinta — alguns são estatísticos, outros comparam períodos (tempo). 
Você pode ativar mais de um método ao mesmo tempo.

---

### 1) IQR — Intervalo Interquartil 📊
**Ideia:** marcar valores muito abaixo/acima da faixa central dos dados.  
**Como funciona:**
1. Calcula Q1 (25%) e Q3 (75%).
2. IQR = Q3 − Q1.
3. Marca como outlier quem estiver **< Q1 − k×IQR** ou **> Q3 + k×IQR**.  

---

### 2) Z-Score — Desvio Padrão 📈
**Ideia:** valores muito distantes da **média** são potenciais outliers.  
**Como funciona:**
1. Calcula média (μ) e desvio padrão (σ).
2. Para cada valor `x`, Z = (x − μ) / σ.

---

### 3) Quantis (Percentis) 📉
**Ideia:** cortar os extremos por percentis específicos.  

---

### 4) MoM — Mês a Mês (Month over Month) 📅
Compara mês atual com o imediatamente anterior.

---

### 5) YoY — Ano a Ano (Year over Year) 📆
Compara com o mesmo mês do ano anterior.

---

### 6) Média Últimos 3 Meses (3M) 📊
**Ideia:** Compara o valor atual com a **média dos últimos 3 meses**.
**Como funciona:**
1. Calcula a média dos valores dos 3 meses anteriores ao mês atual
2. Calcula a variação percentual e absoluta entre o valor atual e essa média
3. Identifica como outlier se a variação ultrapassar o limiar configurado

**Útil para:** Detectar mudanças súbitas em relação à tendência recente de curto prazo.

---

### 7) Média Últimos 6 Meses (6M) 📈
**Ideia:** Compara o valor atual com a **média dos últimos 6 meses**.
**Como funciona:**
1. Calcula a média dos valores dos 6 meses anteriores ao mês atual
2. Calcula a variação percentual e absoluta entre o valor atual e essa média
3. Identifica como outlier se a variação ultrapassar o limiar configurado

**Útil para:** Detectar desvios em relação à tendência de médio prazo, mais estável que a comparação MoM.

---

### 💡 Regra Final
Um valor **só é considerado outlier** quando **(YoY OU MoM OU 3M OU 6M)** **E** **(IQR OU Quantis OU Z-Score)** estiverem verdadeiros ao mesmo tempo.

**Por quê essa combinação?**
- **Métodos temporais** (YoY, MoM, 3M, 6M): Identificam mudanças significativas ao longo do tempo
- **Métodos estatísticos** (IQR, Z-Score, Quantis): Identificam valores extremos na distribuição dos dados
- **Combinação AND**: Garante que apenas valores que são AMBOS atípicos temporalmente E estatisticamente sejam marcados como outliers, reduzindo falsos positivos
"""
    )
