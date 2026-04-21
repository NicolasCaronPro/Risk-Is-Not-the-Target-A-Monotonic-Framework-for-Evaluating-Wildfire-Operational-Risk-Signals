
def fit_spline_mu_classic(df, df_spline=5):
    """
    Fits a spline model Y ~ bs(score) + C(zone) + C(date)
    and returns the predicted mean for each level in LEVELS.
    """
    # Hardcoded LEVELS for safety, matching notebook usage
    LEVELS = [0, 1, 2, 3, 4]

    d = df.dropna(subset=["score", "Y", "zone", "date"]).copy()
    d["score"] = d["score"].clip(0, 4)
    
    # Ensure categorical types for fixed effects
    d["zone"] = d["zone"].astype("category")
    d["date"] = d["date"].astype("category")

    formula = (
        f"Y ~ bs(score, df={df_spline}, degree=3, include_intercept=False, "
        f"lower_bound=0, upper_bound=4) + C(zone) + C(date)"
    )
    
    try:
        fit = smf.ols(formula, data=d).fit(cov_type="HC1")
    except Exception as e:
        # Fallback if too few data points or other issues
        print(f"Warning: Spline fit failed: {e}")
        return {lvl: np.nan for lvl in LEVELS}, np.full(50, np.nan), None

    template = d[["zone", "date"]].copy()
    mu = {}
    
    Y_max = d["Y"].max()
    if np.isnan(Y_max) or Y_max == 0:
        Y_max = 1.0 # arbitrary fallback if Y is all 0 or nan

    for s in LEVELS:
        tmp = template.copy()
        tmp["score"] = s
        pred = fit.predict(tmp)
        
        # Check if clamping was active and warn if significant
        if np.any(pred > Y_max * 100):
             mu[s] = 0
        else:
             pred_clamped = np.clip(pred, 0, Y_max * 2.0)
             mu[s] = pred_clamped.mean()
             
    # Dense evaluation for plotting
    mu_dense = []
    dense_x = np.linspace(0, 4, 50)
    for s in dense_x:
        tmp = template.copy()
        tmp["score"] = s
        pred = fit.predict(tmp)
        if np.any(pred > Y_max * 100):
            mu_dense.append(0.0)
        else:
            pred_clamped = np.clip(pred, 0, Y_max * 2.0)
            mu_dense.append(pred_clamped.mean())
        
    return mu, np.array(mu_dense), fit

def fit_spline_mu(
    df: pd.DataFrame,
    df_spline: int = 6,
    spline_degree: int = 3,
    lambda_curv: float = 10.0,
    dense_points: int = 101,
):
    """
    Fit léger en mémoire :
        Y = intercept + f(score) + FE_zone + FE_date + eps

    avec :
    - f(score) : B-spline de faible rang
    - FE_zone, FE_date : one-hot sparse si informatifs
    - pénalisation de courbure sur les coeffs spline
    - résolution sparse via LSQR

    Règle importante
    ----------------
    Un effet fixe n'est inclus que s'il est informatif, c.-à-d. :
      - il existe plusieurs modalités,
      - mais pas une modalité différente pour chaque observation.

    Donc :
      - si zone est constante -> pas de FE_zone
      - si chaque zone apparaît une seule fois -> pas de FE_zone
      - si date est constante -> pas de FE_date
      - si chaque date apparaît une seule fois -> pas de FE_date

    Paramètres
    ----------
    df : DataFrame avec colonnes ['score', 'Y', 'zone', 'date']
    df_spline : nombre de knots pour la spline
    spline_degree : degré de la spline
    lambda_curv : pénalité sur la dérivée seconde discrète des coeffs spline
    dense_points : nb de points pour mu_dense

    Retourne
    --------
    mu : dict {0: ..., 1: ..., 2: ..., 3: ..., 4: ...}
    mu_dense : np.ndarray
    fit : dict utile pour debug / prédiction
    """
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from scipy.sparse.linalg import lsqr
    from sklearn.preprocessing import OneHotEncoder, SplineTransformer

    req = {"score", "Y", "zone", "date"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {sorted(missing)}")

    work = df[["score", "Y", "zone", "date"]].copy()
    work = work.dropna(subset=["score", "Y", "zone", "date"]).reset_index(drop=True)

    x = work["score"].to_numpy(dtype=np.float64).reshape(-1, 1)
    y = work["Y"].to_numpy(dtype=np.float64)
    n = work.shape[0]

    if n == 0:
        mu = {k: np.nan for k in range(5)}
        return mu, np.full(dense_points, np.nan), {"error": "empty dataframe"}

    # ------------------------------------------------------------------
    # 1) Base spline
    # ------------------------------------------------------------------
    x_clip = np.clip(x, 0.0, 4.0)

    try:
        spline = SplineTransformer(
            n_knots=df_spline,
            degree=spline_degree,
            include_bias=False,
            knots="uniform",
            extrapolation="constant",
            sparse_output=True,
        )
    except TypeError:
        spline = SplineTransformer(
            n_knots=df_spline,
            degree=spline_degree,
            include_bias=False,
            knots="uniform",
            extrapolation="constant",
        )

    B = spline.fit_transform(x_clip)
    B = B.tocsr() if sparse.issparse(B) else sparse.csr_matrix(B)
    n_basis = B.shape[1]

    # ------------------------------------------------------------------
    # 2) Effets fixes conditionnels
    # ------------------------------------------------------------------
    def make_ohe():
        try:
            return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=True)
        except TypeError:
            return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=True)

    n_zone_unique = int(work["zone"].nunique())
    n_date_unique = int(work["date"].nunique())

    # FE informatif <=> plusieurs modalités, mais pas une modalité par observation
    use_zone_fe = (1 < n_zone_unique < n)
    use_date_fe = (1 < n_date_unique < n)

    enc_zone = None
    enc_date = None

    if use_zone_fe:
        enc_zone = make_ohe()
        Z = enc_zone.fit_transform(work[["zone"]])
        Z = Z.tocsr() if sparse.issparse(Z) else sparse.csr_matrix(Z)
    else:
        Z = sparse.csr_matrix((n, 0), dtype=np.float64)

    if use_date_fe:
        enc_date = make_ohe()
        D = enc_date.fit_transform(work[["date"]])
        D = D.tocsr() if sparse.issparse(D) else sparse.csr_matrix(D)
    else:
        D = sparse.csr_matrix((n, 0), dtype=np.float64)
        
    # Intercept sparse
    I = sparse.csr_matrix(np.ones((n, 1), dtype=np.float64))

    # Design matrix sparse
    X = sparse.hstack([I, B, Z, D], format="csr")

    # ------------------------------------------------------------------
    # 3) Pénalité de courbure sur les coeffs spline
    # ------------------------------------------------------------------
    if n_basis >= 3 and lambda_curv > 0:
        D2 = np.diff(np.eye(n_basis), n=2, axis=0)
        n_pen = D2.shape[0]

        P_left = sparse.csr_matrix((n_pen, 1))
        P_mid = sparse.csr_matrix(D2) * np.sqrt(lambda_curv)
        P_zone = sparse.csr_matrix((n_pen, Z.shape[1]))
        P_date = sparse.csr_matrix((n_pen, D.shape[1]))

        P = sparse.hstack([P_left, P_mid, P_zone, P_date], format="csr")

        X_aug = sparse.vstack([X, P], format="csr")
        y_aug = np.concatenate([y, np.zeros(n_pen, dtype=np.float64)])
    else:
        X_aug = X
        y_aug = y

    # ------------------------------------------------------------------
    # 4) Résolution sparse
    # ------------------------------------------------------------------
    sol = lsqr(X_aug, y_aug, atol=1e-8, btol=1e-8, iter_lim=2000)
    beta = sol[0]

    intercept = float(beta[0])
    beta_spline = beta[1:1 + n_basis]
    beta_zone = beta[1:1 + n_basis + Z.shape[1]][n_basis:]
    beta_date = beta[1 + n_basis + Z.shape[1]:]

    # ------------------------------------------------------------------
    # 5) Construire mu(k) et mu_dense
    # ------------------------------------------------------------------
    x_levels = np.arange(5, dtype=np.float64).reshape(-1, 1)
    x_levels_clip = np.clip(x_levels, 0.0, 4.0)

    B_levels = spline.transform(x_levels_clip)
    if sparse.issparse(B_levels):
        B_levels = B_levels.toarray()

    mu_levels = intercept + B_levels @ beta_spline
    mu = {int(k): float(v) for k, v in zip(range(5), mu_levels)}

    x_dense = np.linspace(0.0, 4.0, dense_points, dtype=np.float64).reshape(-1, 1)
    B_dense = spline.transform(x_dense)
    if sparse.issparse(B_dense):
        B_dense = B_dense.toarray()
    mu_dense = intercept + B_dense @ beta_spline

    # ------------------------------------------------------------------
    # 6) Infos utiles
    # ------------------------------------------------------------------
    fit = {
        "intercept": intercept,
        "beta_spline": beta_spline.copy(),
        "beta_zone": beta_zone.copy(),
        "beta_date": beta_date.copy(),
        "spline_transformer": spline,
        "zone_encoder": enc_zone,
        "date_encoder": enc_date,
        "n_obs": int(n),
        "n_basis": int(n_basis),
        "n_zone_unique": int(n_zone_unique),
        "n_date_unique": int(n_date_unique),
        "use_zone_fe": bool(use_zone_fe),
        "use_date_fe": bool(use_date_fe),
        "n_zone_fe": int(Z.shape[1]),
        "n_date_fe": int(D.shape[1]),
        "lambda_curv": float(lambda_curv),
        "lsqr_istop": int(sol[1]),
        "lsqr_iters": int(sol[2]),
        "lsqr_r1norm": float(sol[3]),
        "lsqr_r2norm": float(sol[4]),
    }

    return mu, mu_dense, fit
    
def compute_score_for_k(mu, sigma, k, lvl_counts,
                        min_n=1, min_k=1,
                        w_avg=1.0, w_min=1.0, w_neg=1.0, w_viol=1.0,  # FIXED WEIGHTS
                        min_gain=0.0):
    """
    Computes the score for a given k based on deltas between levels.
    Returns score and coverage (number of valid pairs |P_k*|).
    """
    pairs = PASSAGES.get(k, [])
    deltas = []
    
    coverage = 0

    for (a, b) in pairs:
        # Filter based on min_n
        n_a = lvl_counts.get(a, 0)
        n_b = lvl_counts.get(b, 0)
        
        if n_a >= min_n and n_b >= min_n:
            delta = mu[b] - (mu[a] + min_gain)
            deltas.append(delta)
            coverage += 1.0
            
    # Filter based on min_k
    if coverage < min_k:
        return 0.0, coverage

    if len(deltas) == 0:
        return 0.0, coverage

    deltas = np.array(deltas)
    # Standardize by sigma
    deltas_std = deltas / sigma
    
    # FIXED: Use MEDIAN for average, consistent with comparison_analysis
    avg_delta_std = np.median(deltas_std) 
    
    min_delta_std = np.min(deltas_std)
    neg_mass_std = np.mean(np.clip(-deltas_std, 0.0, None))
    viol_rate = np.mean(deltas_std < 0.0)
    
    score = (
        (w_avg * avg_delta_std + w_min * min_delta_std) / 2
        - w_neg * neg_mass_std * (1 + w_viol * viol_rate)
    )
    
    if np.isnan(score):
        return 0.0, coverage
    
    # 2026-02-13: Clamp score to avoid numerical explosion
    if np.abs(score) > 1e6:
        score = np.clip(score, -1e6, 1e6)

    # Coverage weighting (disabled as per comparison_analysis logic which doesn't seem to force it here)
    return score, coverage


class Scoring:
    """
    Classe regroupant les fonctions de scoring :
      - fit_spline_mu
      - compute_score_for_k
      - evaluation_scoring
      - evaluate_metrics

    Les hyper-paramètres sont fixés à la construction et utilisés par défaut
    dans chaque méthode (mais peuvent être surchargés par appel).

    Normalisation de référence
    --------------------------
    Quand evaluation_scoring(reference=True) est appelé sur un jeu de données
    de référence (ex. entraînement), la classe mémorise la moyenne absolue du
    delta brut pour chaque transition (a→b) : pair_mean_deltas[(a,b)].

    Lors des appels suivants avec reference=False, chaque delta est divisé par
    la moyenne de référence correspondante (au lieu de sigma), permettant une
    comparaison inter-runs sur la même échelle.
    """

    def __init__(
        self,
        df_spline: int = 5,
        min_n: int = 1,
        min_k: int = 0,
        min_gain=None,          # float ou list[float] de longueur 4
        n0: int = 100,
        w_avg: float = 1.0,
        w_min: float = 1.0,
        w_neg: float = 1.0,
        w_viol: float = 1.0
    ):
        self.df_spline = df_spline
        self.min_n     = min_n
        self.min_k     = min_k
        self.min_gain  = min_gain if min_gain is not None else [0.0, 0.0, 0.0, 0.0]
        self.n0        = n0
        self.w_avg     = w_avg
        self.w_min     = w_min
        self.w_neg     = w_neg
        self.w_viol    = w_viol

        # Stocke la moyenne absolue du delta brut par pair (a, b) depuis le run de référence
        self.pair_mean_deltas: dict = {}

    # ------------------------------------------------------------------
    def fit_spline_mu(self, df, df_spline=None):
        """
        Ajuste un spline Y ~ bs(score) + C(zone) + C(date).
        Retourne (mu, mu_dense, fit).
        """
        #return fit_spline_mu(df, df_spline=df_spline if df_spline is not None else self.df_spline)
        return fit_spline_mu(df, df_spline=df_spline if df_spline is not None else self.df_spline)

    def set_sigma(self, sigma):
        self.sigma = sigma

    # ------------------------------------------------------------------
    def _compute_score_for_k_with_ref(self, mu, k, lvl_counts,
                                      min_n, min_k, min_gain,
                                      w_avg, w_min, w_neg, w_viol,
                                      pair_mean_deltas):
        """
        Version interne de compute_score_for_k qui supporte la normalisation
        par pair_mean_deltas.
        Sans pair_mean_deltas : les deltas sont utilisés bruts (pas de division par sigma).

        Retourne (score, coverage, raw_deltas_by_pair)
          - raw_deltas_by_pair : dict {(a,b): delta_brut} pour toutes les paires valides
        """
        pairs = PASSAGES.get(k, [])
        deltas = []
        pairs_used = []
        coverage = 0
        missing_penalized = []  # pairs with missing class but known in reference → penalty -1

        for (a, b) in pairs:
            n_a = lvl_counts.get(a, 0)
            n_b = lvl_counts.get(b, 0)
            if n_a >= min_n and n_b >= min_n:
                delta = mu[b] - (mu[a] + min_gain)
                deltas.append(delta)
                pairs_used.append((a, b))
                coverage += 1

        raw_deltas_by_pair = {pair: d for pair, d in zip(pairs_used, deltas)}
        
        if coverage < min_k or len(deltas) == 0:
            return 0.0, coverage, raw_deltas_by_pair

        deltas = np.array(deltas)

        # ── Normalisation ──────────────────────────────────────────────
        if pair_mean_deltas:
            # Normalise par la valeur de référence de chaque paire
            norm = np.array([
                abs(pair_mean_deltas.get(pair, 1.0)) or 1.0
                for pair in pairs_used
            ])
            deltas_norm = deltas / norm
            #deltas_norm = deltas
        else:
            # Y a été normalisé par sigma avant le spline → deltas déjà en unités σ
            deltas_norm = deltas

        # Injecte les pénalités/neutralités pour les paires sans classe suffisante
        # (-1.0 si paire connue en référence, 0.0 sinon)
        if missing_penalized:
            deltas_norm = np.concatenate([deltas_norm, np.array(missing_penalized)])
        # ──────────────────────────────────────────────────────────────

        avg_delta_std = np.median(deltas_norm)
        min_delta_std = np.min(deltas_norm)
        neg_mass_std  = np.mean(np.clip(-deltas_norm, 0.0, None))
        viol_rate     = np.mean(deltas_norm < 0.0)

        score = (
            (w_avg * avg_delta_std + w_min * min_delta_std) / 2
            - w_neg * neg_mass_std * (1 + w_viol * viol_rate)
        )

        if np.isnan(score):
            return 0.0, coverage, raw_deltas_by_pair
        if np.abs(score) > 1e6:
            score = np.clip(score, -1e6, 1e6)

        return float(score), coverage, raw_deltas_by_pair

    # ------------------------------------------------------------------
    def compute_score_for_k(self, mu, k, lvl_counts,
                            min_n=None, min_k=None, min_gain=None,
                            w_avg=None, w_min=None, w_neg=None, w_viol=None):
        """
        Calcule le score pour un passage k (avec normalisation de référence si disponible).
        Retourne (score, coverage).
        """
        score, coverage, _ = self._compute_score_for_k_with_ref(
            mu, k, lvl_counts,
            min_n    = min_n    if min_n    is not None else self.min_n,
            min_k    = min_k    if min_k    is not None else self.min_k,
            min_gain = min_gain if min_gain is not None else (
                self.min_gain[k - 1] if isinstance(self.min_gain, (list, tuple)) else self.min_gain
            ),
            w_avg  = w_avg  if w_avg  is not None else self.w_avg,
            w_min  = w_min  if w_min  is not None else self.w_min,
            w_neg  = w_neg  if w_neg  is not None else self.w_neg,
            w_viol = w_viol if w_viol is not None else self.w_viol,
            pair_mean_deltas = self.pair_mean_deltas,
        )
        return score, coverage

    # ------------------------------------------------------------------
    def evaluation_scoring(self, ypred, ytrue, dates, zones,
                           df_spline=None, min_n=None, min_k=None,
                           min_gain=None, n0=None,
                           reference: bool = False):
        """
        Lance le scoring monotone complet.
        
        NOTE SUR L'INTERPRÉTATION DU SCORE :
        L'objectif est d'obtenir le score le plus élevé possible, l'idéal étant de se 
        rapprocher de 1.0. Plus le score est proche de 1.0, meilleure est la prédiction monotone.
        Attention : il est extrêmement difficile d'atteindre 1.0. Le score pénalise très fortement 
        les marges négatives et les violations de monotonie inter-classes.
        
        Paramètres
        ----------
        reference : bool (défaut False)
            - True  : mémorise dans self.pair_mean_deltas la moyenne absolue
            - False : si self.pair_mean_deltas est renseigné, normalise les
                      deltas par les moyennes de référence stockées.
                      Sans référence : deltas bruts (pas de division par sigma).

        Retourne (score_high, score_low, coverage_k, score_adj_k, score_min_class, mu, mu_dense).
        """
        _df_spline = df_spline if df_spline is not None else self.df_spline
        _min_n     = min_n     if min_n     is not None else self.min_n
        _min_k     = min_k     if min_k     is not None else self.min_k
        _min_gain  = min_gain  if min_gain  is not None else self.min_gain
        _n0        = n0        if n0        is not None else self.n0

        # ── Spline sur Y normalisé par sigma ────────────────────────────
        df = pd.DataFrame({"score": ypred, "Y": ytrue, "date": dates, "zone": zones})
        df["_lvl"] = df["score"].clip(0, 4).astype(int)
        lvl_counts = df["_lvl"].value_counts().to_dict()

        df["Y"] = df["Y"] / self.sigma
        mu, mu_dense, fit = self.fit_spline_mu(df, df_spline=_df_spline)

        if all(np.isnan(list(mu.values()))):
            return np.nan, np.nan, {}, {}, np.nan, mu, mu_dense

        # ── Calcul des scores (avec ou sans normalisation de référence) ─

        # ── Passe 1 (reference=True) : collecter les deltas bruts directement ──
        if reference:
            raw_pass: dict = {}
            for k in [1, 2, 3, 4]:
                min_g = _min_gain if isinstance(_min_gain, (int, float)) else _min_gain[k - 1]
                for (a, b) in PASSAGES.get(k, []):
                    n_a, n_b = lvl_counts.get(a, 0), lvl_counts.get(b, 0)
                    if n_a >= _min_n and n_b >= _min_n:
                        #print(f'(a, b)=({a}, {b}), (mu[a], mu[b])= {mu[a]}, {mu[b]}')
                        delta = mu[b] - (mu[a] + min_g)
                        raw_pass[(a, b)] = delta

            # Stocker |delta| uniquement pour les paires avec delta POSITIF dans la référence.
            # Les paires anti-monotones (delta < 0) de la référence sont exclues :
            # elles n'ont pas de sens comme étalon de normalisation.
            self.pair_mean_deltas = {
                pair: delta   # delta > 0, donc abs inutile
                for pair, delta in raw_pass.items()
                if not np.isnan(delta) and delta > 1e-9
            }
            
        # ── Passe de scoring : normalise par pair_mean_deltas ────────────────
        # En reference=True  : pair_mean_deltas vient d'être peuplé → normed = ±1 par paire
        # En reference=False : on utilise les valeurs stockées depuis define_reference_model
        pair_mean_deltas_for_scoring = self.pair_mean_deltas

        score_adj_k = {}
        coverage_k  = {}

        for k in [1, 2, 3, 4]:
            min_g = _min_gain if isinstance(_min_gain, (int, float)) else _min_gain[k - 1]

            score, cov, _ = self._compute_score_for_k_with_ref(
                mu, k, lvl_counts,
                min_n    = _min_n,
                min_k    = _min_k,
                min_gain = min_g,
                w_avg    = self.w_avg,
                w_min    = self.w_min,
                w_neg    = self.w_neg,
                w_viol   = self.w_viol,
                pair_mean_deltas = pair_mean_deltas_for_scoring,
            )
            score_adj_k[k] = score
            coverage_k[k]  = cov

        # ── Agrégation ────────────────────────────────────────────────
        score_low  = score_adj_k[1] + score_adj_k[2]
        score_high = score_adj_k[3] + score_adj_k[4]

        score_low  = 0.0 if np.isnan(score_low)  else score_low
        score_high = 0.0 if np.isnan(score_high) else score_high

        if np.sum(ytrue == 0) > _n0:
            c_min = float(np.min(np.round(ypred)))
            score_min_class = -c_min * 2 + 1
        else:
            score_min_class = 0.0

        return score_high, score_low, coverage_k, score_adj_k, score_min_class, mu, mu_dense

    def _plot(self, ypred, ytrue, dates, zones, title="Scoring Fit", dir_output=None):
        """
        Affiche le fit du scoring monotone : mu(score).
        Retourne la figure.
        """
        score_high, score_low, coverage_k, score_adj_k, score_min_class, mu, mu_dense = \
            self.evaluation_scoring(ypred, ytrue, dates, zones)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Dense spline Curve
        if mu_dense is not None and len(mu_dense) > 0:
            x_dense = np.linspace(0.0, 4.0, len(mu_dense))
            ax.plot(x_dense, mu_dense, label='Spline Fit $\mu(score)$', color='blue', linewidth=2, alpha=0.8)

        # Discrete points
        if mu and len(mu) > 0:
            levels = sorted(mu.keys())
            mu_vals = [mu[l] for l in levels]
            ax.scatter(levels, mu_vals, color='red', s=80, label='Discrete Class Means ($\mu_k$)', zorder=5)

            # Annotate mu values
            for l, v in zip(levels, mu_vals):
                ax.annotate(f"{v:.3f}", (l, v), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='red', fontweight='bold')

        # Raw data summary (optional): show average target per level (unadjusted)
        df_raw = pd.DataFrame({"lvl": np.clip(ypred, 0, 4).astype(int), "Y": ytrue})
        # Normalize Y by self.sigma if set
        if hasattr(self, 'sigma') and self.sigma:
            df_raw["Y"] = df_raw["Y"] / self.sigma
            ylabel = "Adjusted Target Mean ($\mu/\sigma$)"
        else:
            ylabel = "Adjusted Target Mean ($\mu$)"

        raw_means = df_raw.groupby("lvl")["Y"].mean()
        ax.scatter(raw_means.index, raw_means.values, marker='x', s=60, color='green', label='Raw Class Means (Unadjusted)', alpha=0.6)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Predicted Class (0-4)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(range(5))
        ax.grid(True, linestyle='--', alpha=0.6)

        # Text box with scores
        scores_text = (f"Score Low: {score_low:.4f}\n"
                       f"Score High: {score_high:.4f}\n"
                       f"Min Class Penalty: {score_min_class:.4f}\n"
                       f"Total: {score_low+score_high+score_min_class:.4f}")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, scores_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        ax.legend(loc='lower right')

        if dir_output:
            out_path = Path(dir_output) / f"{title.lower().replace(' ', '_')}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"Scoring plot saved to {out_path}")

        plt.show()
        return fig

    # ------------------------------------------------------------------
    def evaluate_metrics(self, y_true, y_pred, dates=None, zones=None,
                         y_pred_probas=None, reference: bool = False):
        """
        Calcule l'ensemble des métriques (IoU, F1, recall, score monotone, etc.).
        Le paramètre `reference` est transmis à evaluation_scoring.
        
        NOTE : Pour la métrique 'score' (score monotone), plus la valeur est proche de 1.0, 
        mieux c'est. C'est une métrique très sévère, l'atteinte d'un score de 1.0 est extrêmement difficile.
        
        Retourne un dictionnaire de métriques.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_pred.ndim > 1:
            y_pred = y_pred[:, 0]

        iou   = iou_score(y_true, y_pred)
        f1    = f1_score((y_true > 0).astype(int), (y_pred > 0).astype(int), zero_division=0)
        prec  = precision_score((y_true > 0).astype(int), (y_pred > 0).astype(int), zero_division=0)
        rec   = recall_score((y_true > 0).astype(int), (y_pred > 0).astype(int), zero_division=0)
        f1_macro   = f1_score(y_true.astype(int), y_pred.astype(int), zero_division=0, average='macro')
        prec_macro = precision_score(y_true.astype(int), y_pred.astype(int), zero_division=0, average='macro')
        rec_macro  = recall_score(y_true.astype(int), y_pred.astype(int), zero_division=0, average='macro')
        ent   = entropy(y_pred_probas) if y_pred_probas is not None else 0
        #auoc  = auoc_func(conf_matrix=confusion_matrix(y_true, y_pred, labels=np.union1d(y_true, y_pred)))
        under = under_prediction_score(y_true, y_pred)
        over  = over_prediction_score(y_true, y_pred)

        results = {
            'iou': iou, 'f1': f1, 'under': under, 'over': over,
            'prec': prec, 'recall': rec,
            'auoc': -1.0,
            'f1_macro': f1_macro, 'prec_macro': prec_macro, 'rec_macro': rec_macro,
            'ent': ent,
        }
        
        try:
            if dates is not None and zones is not None:
                score_high, score_low, coverage_k, score_adj_k, score_min_class, mu, mu_dense = \
                    self.evaluation_scoring(y_pred, y_true, dates, zones, reference=reference)
                    
                results['score_high']      = score_high
                results['score_low']       = score_low
                results['score']           = score_high + score_low
                results['score_min_class'] = score_min_class
                for k, count in coverage_k.items():
                    results[f'coverage_k{k}'] = count
                for k, val in score_adj_k.items():
                    results[f'score_k{k}'] = val
                for k, val in mu.items():
                    results[f'mu_{int(k)}'] = val
                for idx, val in enumerate(mu_dense):
                    results[f'mu_dense_{idx}'] = val
            else:
                for key in ('score_high', 'score_low', 'score', 'score_min_class'):
                    results[key] = np.nan
                for k in range(5):
                    results[f'mu_{k}'] = np.nan
        except Exception as e:
            print(f"Warning: monotonic scoring failed in Scoring.evaluate_metrics: {e}")
            for key in ('score_high', 'score_low', 'score', 'score_min_class'):
                results[key] = np.nan
                
        return results
    
    def _plot_matrice(
        self,
        ypred,
        ytrue,
        dates,
        zones,
        title=None,
        dir_output=None,
        df_spline=None,
        min_n=None,
        min_gain=None,
        normalize_with_reference=True,
        show_effective_coverage=True,
        annotate=True,
        figsize=(8, 6),
        vmax_abs=None,
    ):
        """
        Affiche la matrice des transitions Delta_{a->b} pour a < b.

        Chaque case (a, b) de la matrice contient :
        - le delta brut : mu[b] - (mu[a] + min_gain_k)
        - ou le delta normalisé par la référence si normalize_with_reference=True
        - la couverture effective min(n_a, n_b) entre crochets

        Les cases non évaluables (n_a < min_n ou n_b < min_n) sont grisées.

        Paramètres
        ----------
        ypred, ytrue, dates, zones : array-like
            Données nécessaires au calcul monotone.
        title : str or None
            Titre de la figure. Si None, aucun titre n'est affiché.
        dir_output : str or Path or None
            Dossier de sortie pour sauvegarder la figure.
        df_spline, min_n, min_gain :
            Surcharges optionnelles des hyperparamètres de la classe.
        normalize_with_reference : bool
            Si True et si self.pair_mean_deltas est renseigné, chaque delta est divisé
            par la valeur de référence associée à la paire (a, b).
        show_effective_coverage : bool
            Si True, ajoute [min(n_a, n_b)] dans les annotations.
        annotate : bool
            Si True, écrit les valeurs dans les cellules.
        figsize : tuple
            Taille de la figure.
        vmax_abs : float or None
            Borne absolue de la colorbar. Si None, calculée automatiquement.

        Retour
        ------
        fig, ax, delta_plot_mat, delta_raw_mat, coverage_eff_mat, valid_mat
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
        from pathlib import Path

        _df_spline = df_spline if df_spline is not None else self.df_spline
        _min_n = min_n if min_n is not None else self.min_n
        _min_gain = min_gain if min_gain is not None else self.min_gain

        # ------------------------------------------------------------------
        # 1) Calcul des mu(s) via le pipeline existant
        # ------------------------------------------------------------------
        _, _, coverage_k, score_adj_k, _, mu, _ = self.evaluation_scoring(
            ypred, ytrue, dates, zones,
            df_spline=_df_spline,
            min_n=_min_n,
            min_gain=_min_gain,
            reference=False
        )

        if mu is None or len(mu) == 0 or all(np.isnan(list(mu.values()))):
            raise ValueError("Impossible de calculer mu : la spline a échoué ou n'a renvoyé que des NaN.")

        # ------------------------------------------------------------------
        # 2) Comptages par niveau prédit
        # ------------------------------------------------------------------
        ypred = np.asarray(ypred)
        lvl = np.clip(np.round(ypred), 0, 4).astype(int)
        lvl_counts = pd.Series(lvl).value_counts().to_dict()

        n_levels = 5
        delta_raw_mat = np.full((n_levels, n_levels), np.nan, dtype=float)
        delta_plot_mat = np.full((n_levels, n_levels), np.nan, dtype=float)
        coverage_eff_mat = np.full((n_levels, n_levels), np.nan, dtype=float)
        valid_mat = np.zeros((n_levels, n_levels), dtype=bool)

        # ------------------------------------------------------------------
        # 3) Construction des matrices
        # ------------------------------------------------------------------
        for k, pairs in PASSAGES.items():
            min_g = _min_gain if isinstance(_min_gain, (int, float)) else _min_gain[k - 1]

            for (a, b) in pairs:
                n_a = lvl_counts.get(a, 0)
                n_b = lvl_counts.get(b, 0)
                coverage_eff = min(n_a, n_b)

                coverage_eff_mat[a, b] = coverage_eff

                # Paire non évaluable
                if n_a < _min_n or n_b < _min_n:
                    continue

                if a not in mu or b not in mu or np.isnan(mu[a]) or np.isnan(mu[b]):
                    continue

                delta_raw = mu[b] - (mu[a] + min_g)
                delta_raw_mat[a, b] = delta_raw
                valid_mat[a, b] = True

                if normalize_with_reference and getattr(self, "pair_mean_deltas", None):
                    denom = abs(self.pair_mean_deltas.get((a, b), 1.0)) or 1.0
                    delta_plot = delta_raw / denom
                else:
                    delta_plot = delta_raw

                delta_plot_mat[a, b] = delta_plot

        # ------------------------------------------------------------------
        # 4) Préparation du heatmap
        # ------------------------------------------------------------------
        cmap = plt.cm.RdYlGn.copy()
        cmap.set_bad(color="lightgrey")

        valid_values = delta_plot_mat[np.isfinite(delta_plot_mat)]
        if valid_values.size == 0:
            vmax_abs = 1.0 if vmax_abs is None else vmax_abs
        elif vmax_abs is None:
            vmax_abs = float(np.max(np.abs(valid_values)))
            vmax_abs = 1.0 if vmax_abs < 1e-12 else vmax_abs

        norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=vmax_abs)

        masked = np.ma.masked_invalid(delta_plot_mat)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(masked, cmap=cmap, norm=norm)

        # ------------------------------------------------------------------
        # 5) Axes
        # ------------------------------------------------------------------
        ax.set_xticks(range(n_levels))
        ax.set_yticks(range(n_levels))
        ax.set_xticklabels([str(i) for i in range(n_levels)])
        ax.set_yticklabels([str(i) for i in range(n_levels)])
        ax.set_xlabel("To level $b$")
        ax.set_ylabel("From level $a$")

        if title is not None:
            ax.set_title(title)

        # Grille visuelle
        ax.set_xticks(np.arange(-0.5, n_levels, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_levels, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Masquer visuellement la diagonale et le triangle inférieur
        for i in range(n_levels):
            for j in range(n_levels):
                if j <= i:
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1, 1,
                            facecolor="white",
                            edgecolor="white",
                            linewidth=1.0,
                            zorder=3,
                        )
                    )

        # ------------------------------------------------------------------
        # 6) Annotations
        # ------------------------------------------------------------------
        if annotate:
            for a in range(n_levels):
                for b in range(n_levels):
                    if b <= a:
                        continue

                    n_a = lvl_counts.get(a, 0)
                    n_b = lvl_counts.get(b, 0)
                    cov_eff = int(min(n_a, n_b))

                    if valid_mat[a, b]:
                        val = delta_plot_mat[a, b]
                        txt = f"{val:.2f}"
                        if show_effective_coverage:
                            txt += f"\n[{cov_eff}]"

                        # Couleur de texte adaptée au contraste
                        txt_color = "black" if abs(val) < 0.55 * vmax_abs else "white"

                    else:
                        txt = "NA"
                        if show_effective_coverage:
                            txt += f"\n[{cov_eff}]"
                        txt_color = "black"

                    ax.text(
                        b, a, txt,
                        ha="center", va="center",
                        fontsize=10, color=txt_color, zorder=4
                    )

        # ------------------------------------------------------------------
        # 7) Colorbar
        # ------------------------------------------------------------------
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if normalize_with_reference and getattr(self, "pair_mean_deltas", None):
            cbar.set_label(r"Normalized transition effect $\Delta_{a\to b}$")
        else:
            cbar.set_label(r"Transition effect $\Delta_{a\to b}$")

        # ------------------------------------------------------------------
        # 8) Petit résumé en bas
        # ------------------------------------------------------------------
        txt_mode = "normalized by reference" if (normalize_with_reference and getattr(self, "pair_mean_deltas", None)) else "raw"
        extra = (
            f"Mode: {txt_mode}\n"
            f"min_n = {_min_n}\n"
            f"k-scores: "
            + ", ".join([f"k={k}: {score_adj_k.get(k, np.nan):.2f}" for k in [1, 2, 3, 4]])
        )
        """ax.text(
            1.02, 0.02, extra,
            transform=ax.transAxes,
            fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )"""

        fig.tight_layout()

        # ------------------------------------------------------------------
        # 9) Sauvegarde
        # ------------------------------------------------------------------
        if dir_output is not None:
            out_path = Path(dir_output) / f"{(title or 'transition_matrix').lower().replace(' ', '_')}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Transition matrix saved to {out_path}")

        plt.show()

        return fig, ax, delta_plot_mat, delta_raw_mat, coverage_eff_mat, valid_mat
    
    def _plot_fixed_effects(
        self,
        ypred,
        ytrue,
        dates,
        zones,
        df_spline=None,
        top_n_zone=None,
        figsize=None,
        title=None,
        dir_output=None,
    ):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from pathlib import Path

        _df_spline = df_spline if df_spline is not None else self.df_spline

        ypred = np.asarray(ypred)
        ytrue = np.asarray(ytrue)
        dates = np.asarray(dates)
        zones = np.asarray(zones)

        if ypred.ndim > 1:
            ypred = ypred[:, 0]

        if not (len(ypred) == len(ytrue) == len(dates) == len(zones)):
            raise ValueError("ypred, ytrue, dates et zones doivent avoir la même longueur.")

        df = pd.DataFrame({
            "score": ypred,
            "Y": ytrue,
            "date": dates,
            "zone": zones,
        }).dropna(subset=["score", "Y", "date", "zone"]).reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("Aucune observation valide après suppression des NaN.")

        if hasattr(self, "sigma") and self.sigma not in [None, 0]:
            df["Y"] = df["Y"] / self.sigma
            ylabel = "Estimated fixed effect (in sigma units)"
        else:
            ylabel = "Estimated fixed effect"

        mu, mu_dense, fit = self.fit_spline_mu(df, df_spline=_df_spline)

        use_zone_fe = bool(fit.get("use_zone_fe", False))
        use_date_fe = bool(fit.get("use_date_fe", False))

        zone_encoder = fit.get("zone_encoder", None)
        date_encoder = fit.get("date_encoder", None)

        beta_zone = np.asarray(fit.get("beta_zone", []), dtype=float)
        beta_date = np.asarray(fit.get("beta_date", []), dtype=float)

        df_zone = pd.DataFrame(columns=["zone", "effect"])
        if use_zone_fe and zone_encoder is not None:
            zone_cats = list(zone_encoder.categories_[0])
            zone_effects = np.concatenate([[0.0], beta_zone])

            df_zone = pd.DataFrame({
                "zone": zone_cats,
                "effect": zone_effects
            }).sort_values("effect", ascending=True)

            if top_n_zone is not None and top_n_zone < len(df_zone):
                idx = np.argsort(np.abs(df_zone["effect"].values))[-top_n_zone:]
                df_zone = df_zone.iloc[np.sort(idx)].sort_values("effect", ascending=True)

        df_date = pd.DataFrame(columns=["date", "effect"])
        if use_date_fe and date_encoder is not None:
            date_cats = list(date_encoder.categories_[0])
            date_effects = np.concatenate([[0.0], beta_date])

            df_date = pd.DataFrame({
                "date": date_cats,
                "effect": date_effects
            })

            try:
                df_date["date"] = pd.to_datetime(df_date["date"])
                df_date = df_date.sort_values("date")
            except Exception:
                df_date = df_date.sort_values("date")

        n_panels = int(use_zone_fe) + int(use_date_fe)

        if n_panels == 0:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.axis("off")
            ax.text(
                0.5, 0.5,
                "No fixed effect was retained:\n"
                "- either a single zone/date is present,\n"
                "- or one zone/date per observation.",
                ha="center", va="center", fontsize=12
            )
            if title is not None:
                ax.set_title(title)

            fig.tight_layout()
            plt.show()
            return fig, [ax], df_zone, df_date, fit

        # Taille dynamique
        n_zone_labels = len(df_zone) if use_zone_fe else 0
        base_height = 5
        zone_height = max(base_height, 0.38 * max(n_zone_labels, 1))
        final_height = zone_height if use_zone_fe else 5
        final_width = 8 if n_panels == 1 else 15

        if figsize is None:
            figsize = (final_width, final_height)

        fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
        axes = axes.ravel()

        ax_id = 0

        if use_zone_fe:
            ax = axes[ax_id]
            ax.barh(df_zone["zone"].astype(str), df_zone["effect"].values)
            ax.axvline(0.0, linestyle="--", linewidth=1)
            ax.set_xlabel(ylabel)
            ax.set_ylabel("Zone")
            ax.grid(axis="x", linestyle="--", alpha=0.4)

            # Réduit légèrement la taille des labels si beaucoup de zones
            if len(df_zone) > 20:
                ax.tick_params(axis="y", labelsize=9)
            if len(df_zone) > 35:
                ax.tick_params(axis="y", labelsize=8)

            ax_id += 1

        if use_date_fe:
            ax = axes[ax_id]
            x = df_date["date"]
            y = df_date["effect"].values

            ax.plot(x, y, linewidth=1.8)
            ax.axhline(0.0, linestyle="--", linewidth=1)
            ax.set_xlabel("Date")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

            if len(df_date) > 20:
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha("right")

        if title is not None:
            fig.suptitle(title)

        # Marge gauche élargie pour les noms de zones
        if use_zone_fe:
            fig.subplots_adjust(left=0.28, wspace=0.30)
        else:
            fig.tight_layout()

        if dir_output is not None:
            out_path = Path(dir_output) / f"{(title or 'fixed_effects').lower().replace(' ', '_')}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Fixed-effects plot saved to {out_path}")

        plt.show()
        return fig, axes, df_zone, df_date, fit
