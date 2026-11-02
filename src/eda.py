"""Module 3 - Exploratory Data Analysis, heatmaps and zone-level clustering.

Run:  python -m src.eda
Outputs figures to reports/figures and statistics into reports/eda_stats.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

from src import config as C
from src.plot_style import plt, SERIES, SEQ_CMAP, INK2

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SLOT_LABELS = [f"{8 + i // 2:02d}:{30 * (i % 2):02d}" for i in range(C.N_SLOTS)]


def load_grid():
    return pd.read_csv(C.GRID_PARQUET, parse_dates=["slot_time", "date"])


def zone_profiles(grid):
    """Average weekday occupancy curve of each zone (rows=zones, cols=slots)."""
    wk = grid[grid.dow < 5]
    return wk.pivot_table(index="zone", columns="slot", values="rate", aggfunc="mean")


def cluster_zones(prof, k_range=range(2, 7)):
    X = prof.values
    sil = {}
    for k in k_range:
        lab = KMeans(n_clusters=k, n_init=20, random_state=C.RANDOM_STATE).fit_predict(X)
        sil[k] = float(silhouette_score(X, lab))
    best_k = max(sil, key=sil.get)
    km = KMeans(n_clusters=best_k, n_init=20, random_state=C.RANDOM_STATE).fit(X)
    hc = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)
    # order clusters by mean occupancy so cluster 0 = least busy
    order = np.argsort(km.cluster_centers_.mean(axis=1))
    remap = {old: new for new, old in enumerate(order)}
    labels = pd.Series([remap[l] for l in km.labels_], index=prof.index, name="cluster")
    inertia = {k: float(KMeans(n_clusters=k, n_init=20, random_state=C.RANDOM_STATE).fit(X).inertia_)
               for k in k_range}
    return labels, sil, inertia, float(adjusted_rand_score(km.labels_, hc))


def main():
    C.ensure_dirs()
    g = load_grid()
    obs = g.dropna(subset=["rate"])
    stats = {}

    # ---- 1. distribution of occupancy rate --------------------------------
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.hist(obs.rate, bins=40, color=SERIES[0], edgecolor="white", linewidth=0.5)
    for t, lab in [(C.LIMITED_THRESHOLD, "Limited"), (C.FULL_THRESHOLD, "Full")]:
        ax.axvline(t, color=INK2, ls="--", lw=0.8)
        ax.text(t + 0.01, ax.get_ylim()[1] * 0.9, lab, fontsize=6, color=INK2)
    ax.set_xlabel("Occupancy rate"); ax.set_ylabel("Readings")
    ax.set_title("Distribution of occupancy rate")
    fig.savefig(C.FIG_DIR / "eda_rate_hist.png"); plt.close(fig)
    stats["rate_mean"] = float(obs.rate.mean()); stats["rate_median"] = float(obs.rate.median())
    stats["rate_std"] = float(obs.rate.std()); stats["rate_skew"] = float(obs.rate.skew())
    status = pd.cut(obs.rate, [-1, C.LIMITED_THRESHOLD, C.FULL_THRESHOLD, 2], right=False,
                    labels=C.STATUS_LABELS).value_counts(normalize=True)
    stats["status_share"] = {k: round(float(v), 4) for k, v in status.items()}

    # ---- 2. daily curve weekday vs weekend ---------------------------------
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for i, (name, mask) in enumerate([("Weekday", obs.dow < 5), ("Weekend", obs.dow >= 5)]):
        m = obs[mask].groupby("slot").rate.agg(["mean", "std"])
        ax.plot(m.index, m["mean"], color=SERIES[i], label=name)
        ax.fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"], color=SERIES[i], alpha=0.12, lw=0)
    ax.set_xticks(range(0, 18, 3)); ax.set_xticklabels(SLOT_LABELS[::3])
    ax.set_ylabel("Mean occupancy rate"); ax.set_title("Average daily curve (±1 s.d.)")
    ax.legend(loc="upper left")
    fig.savefig(C.FIG_DIR / "eda_daily_curve.png"); plt.close(fig)
    peak = obs.groupby("slot").rate.mean()
    stats["peak_slot"] = SLOT_LABELS[int(peak.idxmax())]; stats["peak_rate"] = float(peak.max())

    # ---- 3. heatmap day-of-week x time ------------------------------------
    hm = obs.pivot_table(index="dow", columns="slot", values="rate", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    im = ax.imshow(hm.values, aspect="auto", cmap=SEQ_CMAP, vmin=0.2, vmax=0.8)
    ax.set_yticks(range(7)); ax.set_yticklabels(DOW)
    ax.set_xticks(range(0, 18, 3)); ax.set_xticklabels(SLOT_LABELS[::3]); ax.grid(False)
    fig.colorbar(im, ax=ax, label="Mean rate", shrink=0.9)
    ax.set_title("Occupancy heatmap: weekday × time")
    fig.savefig(C.FIG_DIR / "eda_heatmap_dow_time.png"); plt.close(fig)
    dmean = obs.groupby("dow").rate.mean()
    stats["busiest_day"] = DOW[int(dmean.idxmax())]; stats["quietest_day"] = DOW[int(dmean.idxmin())]

    # ---- 4. zone x time heatmap + clustering ------------------------------
    prof = zone_profiles(g)
    labels, sil, inertia, ari = cluster_zones(prof)
    stats["cluster_silhouette"] = sil; stats["cluster_inertia"] = inertia
    stats["cluster_k"] = int(labels.nunique()); stats["kmeans_vs_ward_ARI"] = ari
    stats["cluster_sizes"] = labels.value_counts().sort_index().astype(int).to_dict()
    stats["cluster_mean_rate"] = {int(c): float(prof[labels == c].values.mean()) for c in sorted(labels.unique())}
    labels.to_csv(C.PROCESSED_DIR / "zone_clusters.csv")
    order = labels.sort_values().index
    fig, ax = plt.subplots(figsize=(3.4, 4.2))
    im = ax.imshow(prof.loc[order].values, aspect="auto", cmap=SEQ_CMAP, vmin=0, vmax=1)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{z[:16]} (C{labels[z]})" for z in order], fontsize=5)
    ax.set_xticks(range(0, 18, 3)); ax.set_xticklabels(SLOT_LABELS[::3]); ax.grid(False)
    fig.colorbar(im, ax=ax, label="Mean weekday rate", shrink=0.6)
    ax.set_title("Zone-level weekday heatmap (grouped by cluster)")
    fig.savefig(C.FIG_DIR / "eda_zone_heatmap.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.8))
    ks = list(sil)
    axes[0].plot(ks, [inertia[k] for k in ks], marker="o", color=SERIES[0]); axes[0].set_title("Elbow (inertia)")
    axes[1].plot(ks, [sil[k] for k in ks], marker="o", color=SERIES[1]); axes[1].set_title("Silhouette")
    for a in axes: a.set_xlabel("k")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "eda_cluster_selection.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for c in sorted(labels.unique()):
        m = prof[labels == c].mean()
        ax.plot(m.index, m.values, color=SERIES[c], label=f"Cluster {c} (n={int((labels == c).sum())})")
    ax.set_xticks(range(0, 18, 3)); ax.set_xticklabels(SLOT_LABELS[::3]); ax.set_ylim(0, 1)
    ax.set_ylabel("Mean weekday rate"); ax.set_title("Zone cluster centroids"); ax.legend()
    fig.savefig(C.FIG_DIR / "eda_cluster_centroids.png"); plt.close(fig)

    # ---- 5. trend over the period (distribution shift check) ---------------
    daily = obs.groupby("date").rate.mean()
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.plot(daily.index, daily.values, color=SERIES[0])
    for d, lab in [(C.TRAIN_END, "val"), (C.VAL_END, "test")]:
        ax.axvline(pd.Timestamp(d), color=INK2, ls="--", lw=0.8)
        ax.text(pd.Timestamp(d) + pd.Timedelta(days=0.5), daily.max() * 0.98, lab, fontsize=6, color=INK2)
    ax.set_ylabel("Mean daily rate"); ax.set_title("Daily mean occupancy and data split")
    fig.autofmt_xdate(); fig.savefig(C.FIG_DIR / "eda_daily_trend.png"); plt.close(fig)
    split_mean = {
        "train": float(obs[obs.date <= C.TRAIN_END].rate.mean()),
        "val": float(obs[(obs.date > C.TRAIN_END) & (obs.date <= C.VAL_END)].rate.mean()),
        "test": float(obs[obs.date > C.VAL_END].rate.mean()),
    }
    stats["split_mean_rate"] = split_mean

    # ---- 6. missingness & autocorrelation ----------------------------------
    miss = g.groupby("zone").rate.apply(lambda s: s.isna().mean())
    stats["missing_by_zone_max"] = {"zone": miss.idxmax(), "pct": float(miss.max() * 100)}
    stats["missing_overall_pct"] = float(g.rate.isna().mean() * 100)
    s = g.sort_values(["zone", "slot_time"])
    by = s.groupby(["zone", "date"]).rate
    stats["autocorr_lag1slot"] = float(s.rate.corr(by.shift(1)))
    stats["autocorr_lag2slot"] = float(s.rate.corr(by.shift(2)))
    w = s.pivot_table(index=["zone", "date"], columns="slot", values="rate")
    lag1d = w.groupby(level="zone").shift(1); lag7d = w.groupby(level="zone").shift(7)
    stats["corr_same_slot_prev_day"] = float(pd.Series(w.values.ravel()).corr(pd.Series(lag1d.values.ravel())))
    stats["corr_same_slot_prev_week"] = float(pd.Series(w.values.ravel()).corr(pd.Series(lag7d.values.ravel())))
    stats["n_zones"] = int(g.zone.nunique()); stats["n_days"] = int(g.date.nunique())
    stats["capacity_total"] = int(g.groupby("zone").capacity.first().sum())
    stats["capacity_range"] = [int(g.capacity.min()), int(g.capacity.max())]

    (C.ROOT / "reports" / "eda_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
