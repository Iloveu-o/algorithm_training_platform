import os
import numpy as np
import pymysql
# Import plot functions from battery_stats
# Assuming battery_stats.py is in the same directory (backend)
try:
    from battery_stats import plot_features_vs_cycles, plot_features_vs_rul, plot_pcl_distribution, plot_features_heatmap
except ImportError:
    # If import fails (e.g. during refactoring), we might need to handle it or copy functions.
    # But since battery_stats.py exists, it should work.
    pass

def _connect():
    import pymysql
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "12121")
    db = os.getenv("MYSQL_DB", "classdesign")
    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=db,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )

def _fetch_rows(conn, battery_id):
    sql = (
        "SELECT cycle_index, f1, f2, f3, f4, f5, f6, f7, f8, rul, pcl "
        "FROM battery_cycle_data WHERE battery_id=%s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (battery_id,))
        return cur.fetchall()

def _linear_decay(cycles, values):
    cycles = np.asarray(cycles, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(cycles) & np.isfinite(values)
    cycles = cycles[mask]
    values = values[mask]
    if cycles.size < 2 or np.std(values) == 0 or np.std(cycles) == 0:
        return None
    slope, intercept = np.polyfit(cycles, values, 1)
    corr = float(np.corrcoef(cycles, values)[0, 1])
    r2 = corr * corr
    return {"slope": float(slope), "intercept": float(intercept), "corr": corr, "r2": float(r2)}

def _pearson_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or y.size < 2:
        return None
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])

def _stats(arr):
    if arr.size == 0:
        return None
    mean = float(np.mean(arr))
    var = float(np.var(arr))
    std = float(np.std(arr))
    minv = float(np.min(arr))
    maxv = float(np.max(arr))
    
    skew = 0.0
    kurt = 0.0
    if std > 0:
        skew = float(np.mean(((arr - mean) / std) ** 3))
        kurt = float(np.mean(((arr - mean) / std) ** 4) - 3)

    if minv == maxv:
        bins = np.array([minv, maxv])
        counts = np.array([arr.size])
    else:
        counts, bins = np.histogram(arr, bins=10)
    return {
        "count": int(arr.size),
        "mean": mean,
        "var": var,
        "std": std,
        "min": minv,
        "max": maxv,
        "skew": skew,
        "kurtosis": kurt,
        "bins": bins.tolist(),
        "counts": counts.tolist(),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.5)),
        "q75": float(np.quantile(arr, 0.75)),
    }

def analyze_battery(battery_id, create_plots=False):
    try:
        conn = _connect()
    except Exception as e:
        return {"error": f"数据库连接失败：{e}"}
    try:
        rows = _fetch_rows(conn, battery_id)
    except Exception as e:
        conn.close()
        return {"error": f"查询失败：{e}"}
    finally:
        conn.close()

    if not rows:
        return {"error": f"电池组 {battery_id} 无数据", "sample_count": 0}

    features = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
    result = {
        "battery_id": battery_id,
        "sample_count": len(rows),
        "decay": {},
        "features": {},
        "plots": {},
        "raw_data": {
            "cycles": [],
            "rul": [],
            "pcl": [],
            "features": {f: [] for f in features}
        }
    }

    cycles_all = [r["cycle_index"] for r in rows if r["cycle_index"] is not None]
    rul_all = [r["rul"] for r in rows if r["rul"] is not None]
    pcl_all = [r["pcl"] for r in rows if r["pcl"] is not None]
    
    # Fill raw data for frontend plotting
    # Assuming rows are sorted by cycle_index or we should sort them
    # Usually DB returns in insertion order, but better to sort by cycle for plotting
    rows_sorted = sorted([r for r in rows if r["cycle_index"] is not None], key=lambda x: x["cycle_index"])
    
    result["raw_data"]["cycles"] = [r["cycle_index"] for r in rows_sorted]
    result["raw_data"]["rul"] = [r["rul"] for r in rows_sorted]
    result["raw_data"]["pcl"] = [r["pcl"] for r in rows_sorted]
    
    for f in features:
        result["raw_data"]["features"][f] = [r[f] for r in rows_sorted]
    
    decay_rul = _linear_decay(cycles_all, rul_all) if cycles_all and rul_all else None
    decay_pcl = _linear_decay(cycles_all, pcl_all) if cycles_all and pcl_all else None
    
    result["decay"]["rul"] = decay_rul
    result["decay"]["pcl"] = decay_pcl

    for f in features:
        vals = []
        ruls = []
        pcls = []
        cycles = []
        for r in rows:
            v = r.get(f)
            rul_v = r.get("rul")
            pcl_v = r.get("pcl")
            cyc = r.get("cycle_index")
            
            if v is not None:
                vals.append(float(v))
            if v is not None and rul_v is not None:
                ruls.append((float(v), float(rul_v)))
            if v is not None and pcl_v is not None:
                pcls.append((float(v), float(pcl_v)))
            if v is not None and cyc is not None:
                cycles.append((float(cyc), float(v)))
        
        arr = np.array(vals, dtype=float)
        s = _stats(arr)
        
        f_res = {
            "corr_rul": None,
            "corr_pcl": None,
            "decay_cycle": None
        }
        if s:
            f_res.update(s)


        if ruls:
            x_rul = np.array([t[0] for t in ruls])
            y_rul = np.array([t[1] for t in ruls])
            f_res["corr_rul"] = _pearson_corr(x_rul, y_rul)
            
        if pcls:
            x_pcl = np.array([t[0] for t in pcls])
            y_pcl = np.array([t[1] for t in pcls])
            f_res["corr_pcl"] = _pearson_corr(x_pcl, y_pcl)
            
        if cycles:
            cyc = np.array([t[0] for t in cycles])
            val = np.array([t[1] for t in cycles])
            f_res["decay_cycle"] = _linear_decay(cyc, val)
            
        result["features"][f] = f_res

    # Plots (Optional)
    if create_plots:
        try:
            out_path = plot_features_vs_cycles(rows, battery_id)
            result["plots"]["features_vs_cycles"] = os.path.basename(out_path)
        except Exception as e:
            print(f"绘图失败：{e}")

        try:
            out_path_rul = plot_features_vs_rul(rows, battery_id)
            result["plots"]["features_vs_rul"] = os.path.basename(out_path_rul)
        except Exception as e:
            print(f"RUL散点图绘制失败：{e}")

        try:
            out_path_pcl = plot_pcl_distribution(rows, battery_id)
            if out_path_pcl:
                result["plots"]["pcl_distribution"] = os.path.basename(out_path_pcl)
        except Exception as e:
            print(f"PCL分布图绘制失败：{e}")

        try:
            out_path_heatmap = plot_features_heatmap(rows, battery_id)
            if out_path_heatmap:
                result["plots"]["features_heatmap"] = os.path.basename(out_path_heatmap)
        except Exception as e:
            print(f"特征相关性热力图绘制失败：{e}")
        
    return result
