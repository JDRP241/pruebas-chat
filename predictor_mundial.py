# -*- coding: utf-8 -*-
"""
Predictor de partidos internacionales / Mundial 2026
Adaptado de Colab para correr localmente
"""

import pandas as pd
import numpy as np
import requests
import io
import pickle
import json
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

print("Imports listos ✓")

# 45,000 partidos internacionales desde 1872
url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
df_inter = pd.read_csv(url)

# Filtrar solo hasta antes de Qatar 2022
df_inter["date"] = pd.to_datetime(df_inter["date"])
df_train = df_inter[df_inter["date"] < "2022-11-20"].copy()
df_train = df_train.rename(columns={"home_team":"home","away_team":"away","date":"fecha"})
df_train["resultado"] = df_train.apply(
    lambda r: "H" if r.home_score > r.away_score else
              "A" if r.home_score < r.away_score else "D", axis=1
)
df_train = df_train.dropna(subset=["home_score","away_score"])
df_train = df_train.sort_values("fecha").reset_index(drop=True)

print(f"Datos entrenamiento: {len(df_train)} partidos")

# Los 64 partidos de Qatar 2022 como test
df_qatar = df_inter[
    (df_inter["date"] >= "2022-11-20") &
    (df_inter["date"] <= "2022-12-18") &
    (df_inter["tournament"] == "FIFA World Cup")
].copy()
print(f"Partidos Qatar 2022: {len(df_qatar)}")

# Cuotas reales de Qatar 2022 (Bet365, promedio de mercado)
qatar_cuotas = [
    # Grupo A
    {"home": "Qatar",        "away": "Ecuador",       "H": 2.10, "D": 3.20, "A": 3.60, "res": "A"},
    {"home": "Senegal",      "away": "Netherlands",   "H": 4.50, "D": 3.80, "A": 1.75, "res": "A"},
    {"home": "Qatar",        "away": "Senegal",       "H": 2.50, "D": 3.10, "A": 3.00, "res": "A"},
    {"home": "Netherlands",  "away": "Ecuador",       "H": 1.75, "D": 3.50, "A": 4.50, "res": "D"},
    {"home": "Ecuador",      "away": "Senegal",       "H": 2.20, "D": 3.10, "A": 3.40, "res": "A"},
    {"home": "Netherlands",  "away": "Qatar",         "H": 1.30, "D": 5.00, "A": 9.00, "res": "H"},
    # Grupo B
    {"home": "England",      "away": "Iran",          "H": 1.40, "D": 4.50, "A": 7.50, "res": "H"},
    {"home": "United States","away": "Wales",         "H": 2.30, "D": 3.20, "A": 3.10, "res": "D"},
    {"home": "Wales",        "away": "Iran",          "H": 2.40, "D": 3.10, "A": 3.00, "res": "A"},
    {"home": "England",      "away": "United States", "H": 1.75, "D": 3.60, "A": 4.50, "res": "D"},
    {"home": "Wales",        "away": "England",       "H": 5.00, "D": 3.80, "A": 1.65, "res": "A"},
    {"home": "Iran",         "away": "United States", "H": 3.20, "D": 3.10, "A": 2.25, "res": "A"},
    # Grupo C
    {"home": "Argentina",    "away": "Saudi Arabia",  "H": 1.20, "D": 6.50, "A": 14.0, "res": "A"},
    {"home": "Mexico",       "away": "Poland",        "H": 2.40, "D": 3.10, "A": 3.00, "res": "D"},
    {"home": "Poland",       "away": "Saudi Arabia",  "H": 1.90, "D": 3.30, "A": 4.00, "res": "H"},
    {"home": "Argentina",    "away": "Mexico",        "H": 1.55, "D": 3.80, "A": 6.00, "res": "H"},
    {"home": "Poland",       "away": "Argentina",     "H": 4.50, "D": 3.50, "A": 1.80, "res": "A"},
    {"home": "Saudi Arabia", "away": "Mexico",        "H": 4.00, "D": 3.30, "A": 1.90, "res": "A"},
    # Grupo D
    {"home": "Denmark",      "away": "Tunisia",       "H": 1.75, "D": 3.40, "A": 4.50, "res": "D"},
    {"home": "France",       "away": "Australia",     "H": 1.28, "D": 5.50, "A": 10.0, "res": "H"},
    {"home": "Tunisia",      "away": "Australia",     "H": 2.50, "D": 3.10, "A": 2.90, "res": "D"},
    {"home": "France",       "away": "Denmark",       "H": 1.80, "D": 3.40, "A": 4.50, "res": "H"},
    {"home": "Australia",    "away": "Denmark",       "H": 4.20, "D": 3.40, "A": 1.85, "res": "H"},
    {"home": "Tunisia",      "away": "France",        "H": 6.50, "D": 4.20, "A": 1.50, "res": "H"},
    # Grupo E
    {"home": "Spain",        "away": "Costa Rica",    "H": 1.18, "D": 7.00, "A": 14.0, "res": "H"},
    {"home": "Germany",      "away": "Japan",         "H": 1.40, "D": 4.50, "A": 8.00, "res": "A"},
    {"home": "Japan",        "away": "Costa Rica",    "H": 1.65, "D": 3.60, "A": 5.50, "res": "H"},
    {"home": "Spain",        "away": "Germany",       "H": 2.10, "D": 3.20, "A": 3.60, "res": "D"},
    {"home": "Japan",        "away": "Spain",         "H": 5.50, "D": 4.00, "A": 1.60, "res": "H"},
    {"home": "Costa Rica",   "away": "Germany",       "H": 7.00, "D": 4.50, "A": 1.45, "res": "A"},
    # Grupo F
    {"home": "Morocco",      "away": "Croatia",       "H": 3.60, "D": 3.20, "A": 2.10, "res": "D"},
    {"home": "Belgium",      "away": "Canada",        "H": 1.45, "D": 4.20, "A": 7.00, "res": "H"},
    {"home": "Croatia",      "away": "Canada",        "H": 1.75, "D": 3.40, "A": 4.80, "res": "H"},
    {"home": "Morocco",      "away": "Belgium",       "H": 3.80, "D": 3.30, "A": 2.00, "res": "H"},
    {"home": "Croatia",      "away": "Belgium",       "H": 2.80, "D": 3.10, "A": 2.60, "res": "H"},
    {"home": "Canada",       "away": "Morocco",       "H": 3.00, "D": 3.10, "A": 2.40, "res": "A"},
    # Grupo G
    {"home": "Switzerland",  "away": "Cameroon",      "H": 1.75, "D": 3.40, "A": 4.50, "res": "H"},
    {"home": "Brazil",       "away": "Serbia",        "H": 1.40, "D": 4.50, "A": 8.00, "res": "H"},
    {"home": "Cameroon",     "away": "Serbia",        "H": 3.00, "D": 3.10, "A": 2.40, "res": "D"},
    {"home": "Brazil",       "away": "Switzerland",   "H": 1.55, "D": 3.80, "A": 6.00, "res": "H"},
    {"home": "Cameroon",     "away": "Brazil",        "H": 8.00, "D": 5.00, "A": 1.40, "res": "H"},
    {"home": "Serbia",       "away": "Switzerland",   "H": 2.80, "D": 3.10, "A": 2.60, "res": "A"},
    # Grupo H
    {"home": "South Korea",  "away": "Uruguay",       "H": 3.60, "D": 3.20, "A": 2.10, "res": "D"},
    {"home": "Portugal",     "away": "Ghana",         "H": 1.45, "D": 4.20, "A": 7.50, "res": "H"},
    {"home": "Ghana",        "away": "Uruguay",       "H": 3.80, "D": 3.20, "A": 2.00, "res": "A"},
    {"home": "Portugal",     "away": "South Korea",   "H": 1.65, "D": 3.80, "A": 5.00, "res": "H"},
    {"home": "Ghana",        "away": "South Korea",   "H": 3.20, "D": 3.10, "A": 2.25, "res": "A"},
    {"home": "Uruguay",      "away": "Portugal",      "H": 3.50, "D": 3.40, "A": 2.10, "res": "A"},
    # Octavos
    {"home": "Netherlands",  "away": "United States", "H": 1.75, "D": 3.50, "A": 4.50, "res": "H"},
    {"home": "Argentina",    "away": "Australia",     "H": 1.25, "D": 5.50, "A": 11.0, "res": "H"},
    {"home": "France",       "away": "Poland",        "H": 1.45, "D": 4.20, "A": 7.50, "res": "H"},
    {"home": "England",      "away": "Senegal",       "H": 1.55, "D": 3.80, "A": 6.00, "res": "H"},
    {"home": "Japan",        "away": "Croatia",       "H": 2.60, "D": 3.10, "A": 2.80, "res": "A"},
    {"home": "Brazil",       "away": "South Korea",   "H": 1.28, "D": 5.50, "A": 10.0, "res": "H"},
    {"home": "Morocco",      "away": "Spain",         "H": 4.50, "D": 3.50, "A": 1.80, "res": "H"},
    {"home": "Portugal",     "away": "Switzerland",   "H": 1.65, "D": 3.80, "A": 5.00, "res": "H"},
    # Cuartos
    {"home": "Croatia",      "away": "Brazil",        "H": 3.60, "D": 3.20, "A": 2.10, "res": "H"},
    {"home": "Netherlands",  "away": "Argentina",     "H": 3.20, "D": 3.10, "A": 2.25, "res": "A"},
    {"home": "Morocco",      "away": "Portugal",      "H": 4.20, "D": 3.40, "A": 1.85, "res": "H"},
    {"home": "England",      "away": "France",        "H": 2.60, "D": 3.20, "A": 2.80, "res": "A"},
    # Semifinales
    {"home": "Argentina",    "away": "Croatia",       "H": 1.75, "D": 3.50, "A": 4.50, "res": "H"},
    {"home": "France",       "away": "Morocco",       "H": 1.55, "D": 3.80, "A": 6.00, "res": "H"},
    # Tercer puesto
    {"home": "Croatia",      "away": "Morocco",       "H": 2.10, "D": 3.20, "A": 3.60, "res": "H"},
    # Final
    {"home": "Argentina",    "away": "France",        "H": 2.20, "D": 3.40, "A": 3.20, "res": "H"},
]

df_cuotas = pd.DataFrame(qatar_cuotas)
print(f"Partidos con cuotas: {len(df_cuotas)}")

# Funciones de features
def forma_seleccion(df, equipo, fecha, n=10):
    partidos = df[
        ((df["home"]==equipo)|(df["away"]==equipo)) &
        (df["fecha"] < fecha)
    ].tail(n)
    puntos = 0
    for _, row in partidos.iterrows():
        if row["home"]==equipo:
            if row["resultado"]=="H": puntos+=3
            elif row["resultado"]=="D": puntos+=1
        else:
            if row["resultado"]=="A": puntos+=3
            elif row["resultado"]=="D": puntos+=1
    return puntos

def goles_sel(df, equipo, fecha, n=10, favor=True):
    partidos = df[
        ((df["home"]==equipo)|(df["away"]==equipo)) &
        (df["fecha"] < fecha)
    ].tail(n)
    goles = []
    for _, row in partidos.iterrows():
        if row["home"]==equipo:
            goles.append(row["home_score"] if favor else row["away_score"])
        else:
            goles.append(row["away_score"] if favor else row["home_score"])
    return np.mean(goles) if goles else 0

def winrate_sel(df, equipo, fecha, n=20):
    partidos = df[
        ((df["home"]==equipo)|(df["away"]==equipo)) &
        (df["fecha"] < fecha)
    ].tail(n)
    if len(partidos)==0: return 0.5
    wins = 0
    for _, row in partidos.iterrows():
        if row["home"]==equipo and row["resultado"]=="H": wins+=1
        elif row["away"]==equipo and row["resultado"]=="A": wins+=1
    return wins/len(partidos)

print("Funciones listas ✓")
print("Calculando features de entrenamiento... (puede tardar varios minutos)")

df_reciente = df_train[df_train["fecha"] >= "2008-01-01"].copy()
features_train_v2 = []

# Ranking FIFA noviembre 2022
ranking_fifa_2022 = {
    "Brazil": 1, "Belgium": 2, "Argentina": 3, "France": 4,
    "England": 5, "Spain": 7, "Portugal": 8, "Netherlands": 8,
    "Denmark": 10, "Germany": 11, "Mexico": 13, "United States": 13,
    "Croatia": 12, "Uruguay": 14, "Switzerland": 15, "Senegal": 18,
    "Wales": 19, "Poland": 26, "Colombia": 17, "Morocco": 22,
    "Japan": 24, "South Korea": 28, "Australia": 38, "Serbia": 21,
    "Cameroon": 43, "Ecuador": 44, "Ghana": 61, "Tunisia": 30,
    "Canada": 41, "Costa Rica": 31, "Saudi Arabia": 51,
    "Iran": 20, "Qatar": 50,
}

def get_ranking(seleccion, ranking=None):
    if ranking is None:
        ranking = ranking_fifa_2022
    return ranking.get(seleccion, 80)

for _, row in df_reciente.iterrows():
    rank_h = get_ranking(row["home"])
    rank_a = get_ranking(row["away"])
    f = {
        "forma_home":        forma_seleccion(df_train, row["home"], row["fecha"]),
        "forma_away":        forma_seleccion(df_train, row["away"], row["fecha"]),
        "goles_favor_home":  goles_sel(df_train, row["home"], row["fecha"], favor=True),
        "goles_favor_away":  goles_sel(df_train, row["away"], row["fecha"], favor=True),
        "goles_contra_home": goles_sel(df_train, row["home"], row["fecha"], favor=False),
        "goles_contra_away": goles_sel(df_train, row["away"], row["fecha"], favor=False),
        "winrate_home":      winrate_sel(df_train, row["home"], row["fecha"]),
        "winrate_away":      winrate_sel(df_train, row["away"], row["fecha"]),
        "diff_forma":        forma_seleccion(df_train, row["home"], row["fecha"]) -
                             forma_seleccion(df_train, row["away"], row["fecha"]),
        "ranking_home":      rank_h,
        "ranking_away":      rank_a,
        "diff_ranking":      rank_a - rank_h,
        "prob_imp_H":        0.33,
        "prob_imp_D":        0.33,
        "prob_imp_A":        0.33,
        "resultado":         row["resultado"]
    }
    features_train_v2.append(f)

df_feat_wc_v2 = pd.DataFrame(features_train_v2)
print(f"Features listos: {df_feat_wc_v2.shape}")

# Entrenar modelo
le_wc2 = LabelEncoder()
X_wc2 = df_feat_wc_v2.drop("resultado", axis=1)
y_wc2 = le_wc2.fit_transform(df_feat_wc_v2["resultado"])

model_wc2 = XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, eval_metric="mlogloss", random_state=42
)
model_wc2.fit(X_wc2, y_wc2)
print("Modelo entrenado ✓")

# Calcular features Qatar 2022
fecha_mundial = pd.Timestamp("2022-11-20")
features_qatar_v2 = []

for _, row in df_cuotas.iterrows():
    rank_h = get_ranking(row["home"])
    rank_a = get_ranking(row["away"])
    f = {
        "forma_home":        forma_seleccion(df_train, row["home"], fecha_mundial),
        "forma_away":        forma_seleccion(df_train, row["away"], fecha_mundial),
        "goles_favor_home":  goles_sel(df_train, row["home"], fecha_mundial, favor=True),
        "goles_favor_away":  goles_sel(df_train, row["away"], fecha_mundial, favor=True),
        "goles_contra_home": goles_sel(df_train, row["home"], fecha_mundial, favor=False),
        "goles_contra_away": goles_sel(df_train, row["away"], fecha_mundial, favor=False),
        "winrate_home":      winrate_sel(df_train, row["home"], fecha_mundial),
        "winrate_away":      winrate_sel(df_train, row["away"], fecha_mundial),
        "diff_forma":        forma_seleccion(df_train, row["home"], fecha_mundial) -
                             forma_seleccion(df_train, row["away"], fecha_mundial),
        "ranking_home":      rank_h,
        "ranking_away":      rank_a,
        "diff_ranking":      rank_a - rank_h,
        "prob_imp_H":        1/row["H"],
        "prob_imp_D":        1/row["D"],
        "prob_imp_A":        1/row["A"],
    }
    features_qatar_v2.append(f)

df_feat_qatar_v2 = pd.DataFrame(features_qatar_v2)

# Predecir
probs_q2 = model_wc2.predict_proba(df_feat_qatar_v2)
df_cuotas["prob_H2"] = probs_q2[:, 2]
df_cuotas["prob_D2"] = probs_q2[:, 1]
df_cuotas["prob_A2"] = probs_q2[:, 0]
df_cuotas["pred2"]   = le_wc2.inverse_transform(model_wc2.predict(df_feat_qatar_v2))

acc2 = accuracy_score(df_cuotas["res"], df_cuotas["pred2"])
print(f"\nAccuracy Qatar 2022: {acc2:.1%}")

# Simulación value bets
UMBRAL = 0.12
apuesta = 10
resultados_v2 = []

for _, row in df_cuotas.iterrows():
    edge_H = row["prob_H2"] - 1/row["H"]
    edge_D = row["prob_D2"] - 1/row["D"]
    edge_A = row["prob_A2"] - 1/row["A"]
    mejor  = max(edge_H, edge_D, edge_A)

    if mejor > UMBRAL:
        if mejor == edge_H and row["H"] > 2.5:
            apuesta_en = "H"; cuota_ap = row["H"]
        elif mejor == edge_A:
            apuesta_en = "A"; cuota_ap = row["A"]
        elif mejor == edge_D:
            apuesta_en = "D"; cuota_ap = row["D"]
        else:
            continue

        acerto = apuesta_en == row["res"]
        gan    = apuesta*(cuota_ap-1) if acerto else -apuesta
        resultados_v2.append({
            "partido":  f"{row['home']} vs {row['away']}",
            "apostado": apuesta_en,
            "real":     row["res"],
            "cuota":    cuota_ap,
            "acertó":   "SI" if acerto else "NO",
            "ganancia": round(gan, 2)
        })

df_r2 = pd.DataFrame(resultados_v2)
if len(df_r2) > 0:
    roi_v2 = df_r2["ganancia"].sum()/(len(df_r2)*apuesta)*100
    print(f"\nApuestas: {len(df_r2)}")
    print(f"Acertados: {df_r2['acertó'].eq('SI').sum()}/{len(df_r2)}")
    print(f"Ganancia: S/ {df_r2['ganancia'].sum():.2f}")
    print(f"ROI: {roi_v2:.1f}%")
    print()
    print(df_r2.to_string(index=False))

# Guardar modelo localmente
with open("model_mundial_2026.pkl", "wb") as f:
    pickle.dump(model_wc2, f)
with open("le_mundial_2026.pkl", "wb") as f:
    pickle.dump(le_wc2, f)

# Ranking FIFA 2026 actualizado
ranking_fifa_2026 = {
    "Spain": 1, "Argentina": 2, "France": 3, "England": 4,
    "Brazil": 5, "Portugal": 6, "Netherlands": 7, "Morocco": 8,
    "Belgium": 9, "Germany": 10, "Croatia": 11, "Senegal": 12,
    "Italy": 13, "Colombia": 14, "USA": 15, "Mexico": 16,
    "Uruguay": 17, "Switzerland": 18, "Japan": 19, "Iran": 20,
    "Denmark": 21, "South Korea": 22, "Ecuador": 23, "Austria": 24,
    "Turkey": 25, "Nigeria": 26, "Australia": 27, "Algeria": 28,
    "Canada": 29, "Ukraine": 30, "Peru": 38, "Chile": 40,
    "Paraguay": 42, "Venezuela": 45, "Bolivia": 55,
    "Costa Rica": 50, "Panama": 60, "Honduras": 70,
    "Saudi Arabia": 58, "Qatar": 62, "Iraq": 65,
    "Cameroon": 48, "Ghana": 52, "Ivory Coast": 47,
    "South Africa": 56, "Tanzania": 80, "Congo DR": 57,
    "Scotland": 35, "Norway": 36, "Serbia": 33, "Poland": 26,
    "Czech Republic": 37, "Hungary": 39, "Slovakia": 44,
    "New Zealand": 95, "Uzbekistan": 72, "Indonesia": 130,
    # Equipos con nombre distinto al ranking 2022
    "United States": 15,
}

with open("ranking_fifa_2026.json", "w") as f:
    json.dump(ranking_fifa_2026, f, indent=2)

print("\nModelo guardado ✓")

# Función predictor para Mundial 2026
def predecir_partido(home, away, cuota_H, cuota_D, cuota_A, ranking=None):
    if ranking is None:
        ranking = ranking_fifa_2026
    rank_h = get_ranking(home, ranking)
    rank_a = get_ranking(away, ranking)
    fecha_hoy = pd.Timestamp.today()

    features = {
        "forma_home":        forma_seleccion(df_train, home, fecha_hoy),
        "forma_away":        forma_seleccion(df_train, away, fecha_hoy),
        "goles_favor_home":  goles_sel(df_train, home, fecha_hoy, favor=True),
        "goles_favor_away":  goles_sel(df_train, away, fecha_hoy, favor=True),
        "goles_contra_home": goles_sel(df_train, home, fecha_hoy, favor=False),
        "goles_contra_away": goles_sel(df_train, away, fecha_hoy, favor=False),
        "winrate_home":      winrate_sel(df_train, home, fecha_hoy),
        "winrate_away":      winrate_sel(df_train, away, fecha_hoy),
        "diff_forma":        forma_seleccion(df_train, home, fecha_hoy) -
                             forma_seleccion(df_train, away, fecha_hoy),
        "ranking_home":      rank_h,
        "ranking_away":      rank_a,
        "diff_ranking":      rank_a - rank_h,
        "prob_imp_H":        1/cuota_H,
        "prob_imp_D":        1/cuota_D,
        "prob_imp_A":        1/cuota_A,
    }

    X = pd.DataFrame([features])
    probs = model_wc2.predict_proba(X)[0]
    prob_H, prob_D, prob_A = probs[2], probs[1], probs[0]
    edge_H = prob_H - 1/cuota_H
    edge_D = prob_D - 1/cuota_D
    edge_A = prob_A - 1/cuota_A

    print(f"\n{'='*45}")
    print(f"  {home} vs {away}")
    print(f"{'='*45}")
    print(f"  {'Resultado':<12} {'Prob':>8} {'Cuota':>8} {'Edge':>8}")
    print(f"  {'-'*38}")
    print(f"  {'Local (H)':<12} {prob_H*100:>7.1f}%  {cuota_H:>7.2f}  {edge_H:>+7.3f}")
    print(f"  {'Empate (D)':<12} {prob_D*100:>7.1f}%  {cuota_D:>7.2f}  {edge_D:>+7.3f}")
    print(f"  {'Visita (A)':<12} {prob_A*100:>7.1f}%  {cuota_A:>7.2f}  {edge_A:>+7.3f}")
    print(f"{'='*45}")

    mejor = max(edge_H, edge_D, edge_A)
    if mejor > 0.12:
        if mejor == edge_H and cuota_H > 2.5:
            print(f"  >> VALUE BET: Local   cuota {cuota_H}  (edge {mejor:+.3f})")
        elif mejor == edge_A:
            print(f"  >> VALUE BET: Visita  cuota {cuota_A}  (edge {mejor:+.3f})")
        elif mejor == edge_D:
            print(f"  >> VALUE BET: Empate  cuota {cuota_D}  (edge {mejor:+.3f})")
        else:
            print(f"  -- Edge detectado pero filtrado por estrategia")
    else:
        print(f"  -- Sin value bet  (edge max: {mejor:+.3f})")

# Test con partidos del Mundial 2026
print("\n=== PREDICCIONES MUNDIAL 2026 ===")
predecir_partido("Morocco",   "Spain",     4.50, 3.50, 1.80)
predecir_partido("Japan",     "Germany",   4.00, 3.40, 1.90)
predecir_partido("Argentina", "France",    2.20, 3.40, 3.20)
predecir_partido("Colombia",  "Brazil",    3.50, 3.20, 2.10)
