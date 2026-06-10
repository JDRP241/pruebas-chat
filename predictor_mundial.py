# -*- coding: utf-8 -*-
"""
Predictor de partidos internacionales / Mundial 2026
Entrenado con datos hasta junio 2026
"""

import pandas as pd
import numpy as np
import requests
import io
import pickle
import json
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

print("Imports listos ✓")

# Dataset completo de partidos internacionales
url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
df_inter = pd.read_csv(url)
df_inter["date"] = pd.to_datetime(df_inter["date"])

# Entrenar con TODO hasta hoy
hoy = pd.Timestamp.today()
df_train = df_inter[df_inter["date"] < hoy].copy()
df_train = df_train.rename(columns={"home_team":"home","away_team":"away","date":"fecha"})
df_train["resultado"] = df_train.apply(
    lambda r: "H" if r.home_score > r.away_score else
              "A" if r.home_score < r.away_score else "D", axis=1
)
df_train = df_train.dropna(subset=["home_score","away_score"])
df_train = df_train.sort_values("fecha").reset_index(drop=True)

print(f"Total partidos en dataset: {len(df_train)}")
print(f"Rango: {df_train['fecha'].min().date()} → {df_train['fecha'].max().date()}")

# Validación: Qatar 2022
df_cuotas_qatar = [
    {"home": "Qatar",        "away": "Ecuador",       "H": 2.10, "D": 3.20, "A": 3.60, "res": "A"},
    {"home": "Senegal",      "away": "Netherlands",   "H": 4.50, "D": 3.80, "A": 1.75, "res": "A"},
    {"home": "Qatar",        "away": "Senegal",       "H": 2.50, "D": 3.10, "A": 3.00, "res": "A"},
    {"home": "Netherlands",  "away": "Ecuador",       "H": 1.75, "D": 3.50, "A": 4.50, "res": "D"},
    {"home": "Ecuador",      "away": "Senegal",       "H": 2.20, "D": 3.10, "A": 3.40, "res": "A"},
    {"home": "Netherlands",  "away": "Qatar",         "H": 1.30, "D": 5.00, "A": 9.00, "res": "H"},
    {"home": "England",      "away": "Iran",          "H": 1.40, "D": 4.50, "A": 7.50, "res": "H"},
    {"home": "United States","away": "Wales",         "H": 2.30, "D": 3.20, "A": 3.10, "res": "D"},
    {"home": "Wales",        "away": "Iran",          "H": 2.40, "D": 3.10, "A": 3.00, "res": "A"},
    {"home": "England",      "away": "United States", "H": 1.75, "D": 3.60, "A": 4.50, "res": "D"},
    {"home": "Wales",        "away": "England",       "H": 5.00, "D": 3.80, "A": 1.65, "res": "A"},
    {"home": "Iran",         "away": "United States", "H": 3.20, "D": 3.10, "A": 2.25, "res": "A"},
    {"home": "Argentina",    "away": "Saudi Arabia",  "H": 1.20, "D": 6.50, "A": 14.0, "res": "A"},
    {"home": "Mexico",       "away": "Poland",        "H": 2.40, "D": 3.10, "A": 3.00, "res": "D"},
    {"home": "Poland",       "away": "Saudi Arabia",  "H": 1.90, "D": 3.30, "A": 4.00, "res": "H"},
    {"home": "Argentina",    "away": "Mexico",        "H": 1.55, "D": 3.80, "A": 6.00, "res": "H"},
    {"home": "Poland",       "away": "Argentina",     "H": 4.50, "D": 3.50, "A": 1.80, "res": "A"},
    {"home": "Saudi Arabia", "away": "Mexico",        "H": 4.00, "D": 3.30, "A": 1.90, "res": "A"},
    {"home": "Denmark",      "away": "Tunisia",       "H": 1.75, "D": 3.40, "A": 4.50, "res": "D"},
    {"home": "France",       "away": "Australia",     "H": 1.28, "D": 5.50, "A": 10.0, "res": "H"},
    {"home": "Tunisia",      "away": "Australia",     "H": 2.50, "D": 3.10, "A": 2.90, "res": "D"},
    {"home": "France",       "away": "Denmark",       "H": 1.80, "D": 3.40, "A": 4.50, "res": "H"},
    {"home": "Australia",    "away": "Denmark",       "H": 4.20, "D": 3.40, "A": 1.85, "res": "H"},
    {"home": "Tunisia",      "away": "France",        "H": 6.50, "D": 4.20, "A": 1.50, "res": "H"},
    {"home": "Spain",        "away": "Costa Rica",    "H": 1.18, "D": 7.00, "A": 14.0, "res": "H"},
    {"home": "Germany",      "away": "Japan",         "H": 1.40, "D": 4.50, "A": 8.00, "res": "A"},
    {"home": "Japan",        "away": "Costa Rica",    "H": 1.65, "D": 3.60, "A": 5.50, "res": "H"},
    {"home": "Spain",        "away": "Germany",       "H": 2.10, "D": 3.20, "A": 3.60, "res": "D"},
    {"home": "Japan",        "away": "Spain",         "H": 5.50, "D": 4.00, "A": 1.60, "res": "H"},
    {"home": "Costa Rica",   "away": "Germany",       "H": 7.00, "D": 4.50, "A": 1.45, "res": "A"},
    {"home": "Morocco",      "away": "Croatia",       "H": 3.60, "D": 3.20, "A": 2.10, "res": "D"},
    {"home": "Belgium",      "away": "Canada",        "H": 1.45, "D": 4.20, "A": 7.00, "res": "H"},
    {"home": "Croatia",      "away": "Canada",        "H": 1.75, "D": 3.40, "A": 4.80, "res": "H"},
    {"home": "Morocco",      "away": "Belgium",       "H": 3.80, "D": 3.30, "A": 2.00, "res": "H"},
    {"home": "Croatia",      "away": "Belgium",       "H": 2.80, "D": 3.10, "A": 2.60, "res": "H"},
    {"home": "Canada",       "away": "Morocco",       "H": 3.00, "D": 3.10, "A": 2.40, "res": "A"},
    {"home": "Switzerland",  "away": "Cameroon",      "H": 1.75, "D": 3.40, "A": 4.50, "res": "H"},
    {"home": "Brazil",       "away": "Serbia",        "H": 1.40, "D": 4.50, "A": 8.00, "res": "H"},
    {"home": "Cameroon",     "away": "Serbia",        "H": 3.00, "D": 3.10, "A": 2.40, "res": "D"},
    {"home": "Brazil",       "away": "Switzerland",   "H": 1.55, "D": 3.80, "A": 6.00, "res": "H"},
    {"home": "Cameroon",     "away": "Brazil",        "H": 8.00, "D": 5.00, "A": 1.40, "res": "H"},
    {"home": "Serbia",       "away": "Switzerland",   "H": 2.80, "D": 3.10, "A": 2.60, "res": "A"},
    {"home": "South Korea",  "away": "Uruguay",       "H": 3.60, "D": 3.20, "A": 2.10, "res": "D"},
    {"home": "Portugal",     "away": "Ghana",         "H": 1.45, "D": 4.20, "A": 7.50, "res": "H"},
    {"home": "Ghana",        "away": "Uruguay",       "H": 3.80, "D": 3.20, "A": 2.00, "res": "A"},
    {"home": "Portugal",     "away": "South Korea",   "H": 1.65, "D": 3.80, "A": 5.00, "res": "H"},
    {"home": "Ghana",        "away": "South Korea",   "H": 3.20, "D": 3.10, "A": 2.25, "res": "A"},
    {"home": "Uruguay",      "away": "Portugal",      "H": 3.50, "D": 3.40, "A": 2.10, "res": "A"},
    {"home": "Netherlands",  "away": "United States", "H": 1.75, "D": 3.50, "A": 4.50, "res": "H"},
    {"home": "Argentina",    "away": "Australia",     "H": 1.25, "D": 5.50, "A": 11.0, "res": "H"},
    {"home": "France",       "away": "Poland",        "H": 1.45, "D": 4.20, "A": 7.50, "res": "H"},
    {"home": "England",      "away": "Senegal",       "H": 1.55, "D": 3.80, "A": 6.00, "res": "H"},
    {"home": "Japan",        "away": "Croatia",       "H": 2.60, "D": 3.10, "A": 2.80, "res": "A"},
    {"home": "Brazil",       "away": "South Korea",   "H": 1.28, "D": 5.50, "A": 10.0, "res": "H"},
    {"home": "Morocco",      "away": "Spain",         "H": 4.50, "D": 3.50, "A": 1.80, "res": "H"},
    {"home": "Portugal",     "away": "Switzerland",   "H": 1.65, "D": 3.80, "A": 5.00, "res": "H"},
    {"home": "Croatia",      "away": "Brazil",        "H": 3.60, "D": 3.20, "A": 2.10, "res": "H"},
    {"home": "Netherlands",  "away": "Argentina",     "H": 3.20, "D": 3.10, "A": 2.25, "res": "A"},
    {"home": "Morocco",      "away": "Portugal",      "H": 4.20, "D": 3.40, "A": 1.85, "res": "H"},
    {"home": "England",      "away": "France",        "H": 2.60, "D": 3.20, "A": 2.80, "res": "A"},
    {"home": "Argentina",    "away": "Croatia",       "H": 1.75, "D": 3.50, "A": 4.50, "res": "H"},
    {"home": "France",       "away": "Morocco",       "H": 1.55, "D": 3.80, "A": 6.00, "res": "H"},
    {"home": "Croatia",      "away": "Morocco",       "H": 2.10, "D": 3.20, "A": 3.60, "res": "H"},
    {"home": "Argentina",    "away": "France",        "H": 2.20, "D": 3.40, "A": 3.20, "res": "H"},
]
df_cuotas = pd.DataFrame(df_cuotas_qatar)

# Ranking FIFA junio 2026 (actualizado)
ranking_fifa_2026 = {
    "Spain": 1, "Argentina": 2, "France": 3, "England": 4,
    "Brazil": 5, "Portugal": 6, "Netherlands": 7, "Morocco": 8,
    "Belgium": 9, "Germany": 10, "Croatia": 11, "Senegal": 12,
    "Italy": 13, "Colombia": 14, "United States": 15, "Mexico": 16,
    "Uruguay": 17, "Switzerland": 18, "Japan": 19, "Iran": 20,
    "Denmark": 21, "South Korea": 22, "Ecuador": 23, "Austria": 24,
    "Turkey": 25, "Nigeria": 26, "Australia": 27, "Algeria": 28,
    "Canada": 29, "Ukraine": 30, "Scotland": 35, "Norway": 36,
    "Czech Republic": 37, "Peru": 38, "Hungary": 39, "Chile": 40,
    "Serbia": 33, "Poland": 26, "Slovakia": 44, "Venezuela": 45,
    "Ivory Coast": 47, "Cameroon": 48, "Costa Rica": 50,
    "Ghana": 52, "South Africa": 56, "Congo DR": 57,
    "Saudi Arabia": 58, "Qatar": 62, "Iraq": 65, "Honduras": 70,
    "Uzbekistan": 72, "Tanzania": 80, "Bolivia": 55,
    "Paraguay": 42, "Panama": 60, "Indonesia": 130,
    "New Zealand": 95, "Wales": 19, "Tunisia": 30,
    # Aliases
    "USA": 15,
}

with open("ranking_fifa_2026.json", "w") as f:
    json.dump(ranking_fifa_2026, f, indent=2)

def get_ranking(seleccion, ranking=None):
    if ranking is None:
        ranking = ranking_fifa_2026
    return ranking.get(seleccion, 80)

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
print("Calculando features de entrenamiento con datos hasta 2026... (puede tardar varios minutos)")

# Usar datos desde 2010 para capturar era moderna del fútbol
df_reciente = df_train[df_train["fecha"] >= "2010-01-01"].copy()
features_train = []

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
    features_train.append(f)

df_feat = pd.DataFrame(features_train)
print(f"Features listos: {df_feat.shape}")

# Entrenar modelo
le = LabelEncoder()
X = df_feat.drop("resultado", axis=1)
y = le.fit_transform(df_feat["resultado"])

model = XGBClassifier(
    n_estimators=600, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", random_state=42
)
model.fit(X, y)
print("Modelo entrenado ✓")

# Validar con Qatar 2022
print("\nValidando con Qatar 2022...")
fecha_qatar = pd.Timestamp("2022-11-20")
df_train_pre_qatar = df_train[df_train["fecha"] < fecha_qatar]

feats_val = []
for _, row in df_cuotas.iterrows():
    rank_h = get_ranking(row["home"])
    rank_a = get_ranking(row["away"])
    f = {
        "forma_home":        forma_seleccion(df_train_pre_qatar, row["home"], fecha_qatar),
        "forma_away":        forma_seleccion(df_train_pre_qatar, row["away"], fecha_qatar),
        "goles_favor_home":  goles_sel(df_train_pre_qatar, row["home"], fecha_qatar, favor=True),
        "goles_favor_away":  goles_sel(df_train_pre_qatar, row["away"], fecha_qatar, favor=True),
        "goles_contra_home": goles_sel(df_train_pre_qatar, row["home"], fecha_qatar, favor=False),
        "goles_contra_away": goles_sel(df_train_pre_qatar, row["away"], fecha_qatar, favor=False),
        "winrate_home":      winrate_sel(df_train_pre_qatar, row["home"], fecha_qatar),
        "winrate_away":      winrate_sel(df_train_pre_qatar, row["away"], fecha_qatar),
        "diff_forma":        forma_seleccion(df_train_pre_qatar, row["home"], fecha_qatar) -
                             forma_seleccion(df_train_pre_qatar, row["away"], fecha_qatar),
        "ranking_home":      rank_h,
        "ranking_away":      rank_a,
        "diff_ranking":      rank_a - rank_h,
        "prob_imp_H":        1/row["H"],
        "prob_imp_D":        1/row["D"],
        "prob_imp_A":        1/row["A"],
    }
    feats_val.append(f)

df_val = pd.DataFrame(feats_val)
probs_val = model.predict_proba(df_val)
df_cuotas["prob_H"] = probs_val[:, 2]
df_cuotas["prob_D"] = probs_val[:, 1]
df_cuotas["prob_A"] = probs_val[:, 0]
df_cuotas["pred"]   = le.inverse_transform(model.predict(df_val))

acc = accuracy_score(df_cuotas["res"], df_cuotas["pred"])
print(f"Accuracy Qatar 2022: {acc:.1%}")

# Guardar modelo
with open("model_mundial_2026.pkl", "wb") as f:
    pickle.dump(model, f)
with open("le_mundial_2026.pkl", "wb") as f:
    pickle.dump(le, f)
print("Modelo guardado ✓")

# Función principal de predicción
def predecir_partido(home, away, cuota_H, cuota_D, cuota_A):
    rank_h = get_ranking(home)
    rank_a = get_ranking(away)
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

    Xp = pd.DataFrame([features])
    probs = model.predict_proba(Xp)[0]
    prob_H, prob_D, prob_A = probs[2], probs[1], probs[0]
    edge_H = prob_H - 1/cuota_H
    edge_D = prob_D - 1/cuota_D
    edge_A = prob_A - 1/cuota_A

    print(f"\n{'='*48}")
    print(f"  {home} vs {away}")
    print(f"  Ranking FIFA: {rank_h} vs {rank_a}")
    print(f"{'='*48}")
    print(f"  {'Resultado':<12} {'Prob%':>7} {'Cuota':>7} {'Edge':>8}")
    print(f"  {'-'*40}")
    print(f"  {'Local (H)':<12} {prob_H*100:>6.1f}%  {cuota_H:>6.2f}  {edge_H:>+7.3f}")
    print(f"  {'Empate (D)':<12} {prob_D*100:>6.1f}%  {cuota_D:>6.2f}  {edge_D:>+7.3f}")
    print(f"  {'Visita (A)':<12} {prob_A*100:>6.1f}%  {cuota_A:>6.2f}  {edge_A:>+7.3f}")
    print(f"{'='*48}")

    mejor = max(edge_H, edge_D, edge_A)
    if mejor > 0.12:
        if mejor == edge_H and cuota_H > 2.5:
            print(f"  >> VALUE BET: Local   @{cuota_H}  (edge {mejor:+.3f})")
        elif mejor == edge_A:
            print(f"  >> VALUE BET: Visita  @{cuota_A}  (edge {mejor:+.3f})")
        elif mejor == edge_D:
            print(f"  >> VALUE BET: Empate  @{cuota_D}  (edge {mejor:+.3f})")
        else:
            print(f"  -- Edge existe pero no pasa filtro de cuota >2.5")
    elif mejor > 0.05:
        print(f"  ~ Edge débil: {mejor:+.3f} — no recomendado")
    else:
        print(f"  -- Sin value bet  (edge max: {mejor:+.3f})")

# ============================================================
# PARTIDO DE HOY — 10 junio 2026
# Amistoso internacional: Mexico vs Ecuador (ciudad de Mexico)
# Cuotas aproximadas de mercado
# ============================================================
print("\n" + "="*48)
print("  PARTIDO DE HOY — 10 junio 2026")
print("  Amistoso previo al Mundial")
print("="*48)
predecir_partido("Mexico", "Ecuador", 2.20, 3.10, 3.50)

# Bonus: primer partido del Mundial 2026 (mañana 11 jun)
print("\n" + "="*48)
print("  APERTURA MUNDIAL 2026 — 11 junio")
print("  Grupo A: Mexico vs (sede USA)")
print("="*48)
predecir_partido("Mexico",        "United States", 3.20, 3.10, 2.30)
predecir_partido("Canada",        "Colombia",      3.80, 3.30, 2.10)
