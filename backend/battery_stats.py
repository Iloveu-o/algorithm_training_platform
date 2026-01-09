import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

def _require_pymysql():
    try:
        import pymysql  # noqa: F401
    except Exception as e:
        print("缺少pymysql依赖，请先安装：pip install pymysql")
        raise

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
    minv = float(np.min(arr))
    maxv = float(np.max(arr))
    if minv == maxv:
        bins = np.array([minv, maxv])
        counts = np.array([arr.size])
    else:
        counts, bins = np.histogram(arr, bins=10)
    return {
        "count": int(arr.size),
        "mean": mean,
        "var": var,
        "min": minv,
        "max": maxv,
        "bins": bins.tolist(),
        "counts": counts.tolist(),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.5)),
        "q75": float(np.quantile(arr, 0.75)),
    }

def plot_features_vs_cycles(rows, battery_id):
    features = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), constrained_layout=True)
    axes = axes.flatten()
    for i, f in enumerate(features):
        pairs = [(r["cycle_index"], r[f]) for r in rows if r["cycle_index"] is not None and r[f] is not None]
        if not pairs:
            axes[i].set_title(f"{f} 无数据")
            axes[i].set_xlabel("cycle_index")
            axes[i].set_ylabel(f)
            continue
        pairs.sort(key=lambda x: x[0])
        xs = [float(p[0]) for p in pairs]
        ys = [float(p[1]) for p in pairs]
        axes[i].plot(xs, ys, color="#1f77b4", linewidth=1.5)
        axes[i].scatter(xs, ys, s=10, color="#1f77b4")
        axes[i].set_title(f"{f} 随循环次数变化")
        axes[i].set_xlabel("cycle_index")
        axes[i].set_ylabel(f)
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", f"battery_{battery_id}_features_vs_cycles.png")
    plt.savefig(out_path, dpi=120)
    return out_path

def plot_features_vs_rul(rows, battery_id):
    features = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), constrained_layout=True)
    axes = axes.flatten()
    for i, f in enumerate(features):
        pairs = [(r["rul"], r[f]) for r in rows if r["rul"] is not None and r[f] is not None]
        if not pairs:
            axes[i].set_title(f"{f} 无数据")
            axes[i].set_xlabel("RUL")
            axes[i].set_ylabel(f)
            continue
        # RUL通常是倒序的，这里不需要排序，直接画散点即可
        xs = [float(p[0]) for p in pairs]
        ys = [float(p[1]) for p in pairs]
        # 绘制散点
        axes[i].scatter(xs, ys, s=10, color="#ff7f0e", alpha=0.6)
        axes[i].set_title(f"{f} 与 RUL 的关系")
        axes[i].set_xlabel("RUL")
        axes[i].set_ylabel(f)
        # 翻转X轴，符合RUL从大到小（寿命从开始到结束）的直观感觉，或者保持原样
        # 既然是相关性散点图，通常保持坐标轴数值递增。如果想表达随时间变化，X轴应该是cycle。
        # 这里X轴是RUL，RUL越小表示越接近失效。保持默认递增坐标轴即可。

    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", f"battery_{battery_id}_features_vs_rul.png")
    plt.savefig(out_path, dpi=120)
    return out_path

def plot_pcl_distribution(rows, battery_id):
    pcls = [r["pcl"] for r in rows if r["pcl"] is not None]
    if not pcls:
        return None
    
    plt.figure(figsize=(8, 6))
    plt.hist(pcls, bins=20, color="#2ca02c", edgecolor="black", alpha=0.7)
    plt.title(f"电池 {battery_id} PCL (预测容量损失) 分布")
    plt.xlabel("PCL")
    plt.ylabel("频数")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", f"battery_{battery_id}_pcl_dist.png")
    plt.savefig(out_path, dpi=120)
    return out_path

def plot_features_heatmap(rows, battery_id):
    features = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
    data_matrix = []
    
    # 提取每一行的数据，构建矩阵
    for r in rows:
        row_data = []
        has_none = False
        for f in features:
            val = r.get(f)
            if val is None:
                has_none = True
                break
            row_data.append(float(val))
        if not has_none:
            data_matrix.append(row_data)
            
    if not data_matrix:
        return None
        
    data_matrix = np.array(data_matrix)
    # 计算相关系数矩阵
    # rowvar=False 表示每一列是一个变量（特征）
    if data_matrix.shape[0] < 2:
        return None
        
    corr_matrix = np.corrcoef(data_matrix, rowvar=False)
    
    # 检查相关矩阵是否包含NaN（标准差为0的情况）
    if np.isnan(corr_matrix).all():
        return None
        
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("皮尔逊相关系数", rotation=-90, va="bottom")
    
    # 设置刻度标签
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(features)
    ax.set_yticklabels(features)
    
    # 让x轴标签显示在顶部
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    
    # 在每个格子里显示数值
    for i in range(len(features)):
        for j in range(len(features)):
            val = corr_matrix[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")
                
    ax.set_title(f"电池 {battery_id} 特征相关性热力图", y=-0.1)
    fig.tight_layout()
    
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", f"battery_{battery_id}_features_heatmap.png")
    plt.savefig(out_path, dpi=120)
    return out_path

def main():
    _require_pymysql()
    try:
        battery_id_str = input("请输入电池组编号 battery_id：").strip()
        battery_id = int(battery_id_str)
    except Exception:
        print("battery_id输入不合法，请输入整数。")
        return
    try:
        conn = _connect()
    except Exception as e:
        print(f"数据库连接失败：{e}")
        return
    try:
        rows = _fetch_rows(conn, battery_id)
    except Exception as e:
        print(f"查询失败：{e}")
        conn.close()
        return
    finally:
        conn.close()
    features = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
    print(f"电池组 {battery_id} 样本行数：{len(rows)}")
    cycles_all = [r["cycle_index"] for r in rows if r["cycle_index"] is not None]
    rul_all = [r["rul"] for r in rows if r["rul"] is not None]
    pcl_all = [r["pcl"] for r in rows if r["pcl"] is not None]
    decay_rul = _linear_decay(cycles_all, rul_all) if cycles_all and rul_all else None
    decay_pcl = _linear_decay(cycles_all, pcl_all) if cycles_all and pcl_all else None
    print("\nRUL-循环衰减：")
    if decay_rul:
        print(f"相关性: {decay_rul['corr']}, 斜率/每循环: {decay_rul['slope']}, R2: {decay_rul['r2']}")
    else:
        print("无数据或无效")
    print("\nPCL-循环衰减：")
    if decay_pcl:
        print(f"相关性: {decay_pcl['corr']}, 斜率/每循环: {decay_pcl['slope']}, R2: {decay_pcl['r2']}")
    else:
        print("无数据或无效")
    for f in features:
        vals = []
        ruls = []
        pcls = []
        cycles = []
        for r in rows:
            v = r[f]
            rul_v = r["rul"]
            pcl_v = r["pcl"]
            cyc = r["cycle_index"]
            if v is not None:
                vals.append(v)
            if v is not None and rul_v is not None:
                ruls.append((float(v), float(rul_v)))
            if v is not None and pcl_v is not None:
                pcls.append((float(v), float(pcl_v)))
            if v is not None and cyc is not None:
                cycles.append((float(cyc), float(v)))
        arr = np.array(vals, dtype=float)
        s = _stats(arr)
        print(f"\n特征 {f}：")
        if s is None:
            print("无数据")
            continue
        print(f"数量: {s['count']}")
        print(f"均值: {s['mean']}")
        print(f"方差: {s['var']}")
        print(f"最小值: {s['min']}")
        print(f"最大值: {s['max']}")
        print(f"分布-分位数: Q25={s['q25']}, 中位数={s['median']}, Q75={s['q75']}")
        print(f"分布-直方图bins: {s['bins']}")
        print(f"分布-直方图counts: {s['counts']}")
        if ruls:
            x_rul = np.array([t[0] for t in ruls])
            y_rul = np.array([t[1] for t in ruls])
            corr_rul = _pearson_corr(x_rul, y_rul)
            print(f"与RUL的皮尔逊相关系数: {corr_rul if corr_rul is not None else '无效/常数序列'}")
        else:
            print("与RUL的皮尔逊相关系数: 无数据")
        if pcls:
            x_pcl = np.array([t[0] for t in pcls])
            y_pcl = np.array([t[1] for t in pcls])
            corr_pcl = _pearson_corr(x_pcl, y_pcl)
            print(f"与PCL的皮尔逊相关系数: {corr_pcl if corr_pcl is not None else '无效/常数序列'}")
        else:
            print("与PCL的皮尔逊相关系数: 无数据")
        if cycles:
            cyc = np.array([t[0] for t in cycles])
            val = np.array([t[1] for t in cycles])
            decay = _linear_decay(cyc, val)
            if decay:
                print(f"与循环次数相关性: {decay['corr']}, 斜率/每循环: {decay['slope']}, R2: {decay['r2']}")
            else:
                print("与循环次数相关性: 无数据或无效")
    try:
        out_path = plot_features_vs_cycles(rows, battery_id)
        print(f"\n已保存折线图: {out_path}")
    except Exception as e:
        print(f"\n绘图失败：{e}")

    try:
        out_path_rul = plot_features_vs_rul(rows, battery_id)
        print(f"已保存散点图: {out_path_rul}")
    except Exception as e:
        print(f"RUL散点图绘制失败：{e}")

    try:
        out_path_pcl = plot_pcl_distribution(rows, battery_id)
        if out_path_pcl:
            print(f"已保存PCL分布图: {out_path_pcl}")
        else:
            print("PCL数据为空，未生成分布图")
    except Exception as e:
        print(f"PCL分布图绘制失败：{e}")

    try:
        out_path_heatmap = plot_features_heatmap(rows, battery_id)
        if out_path_heatmap:
            print(f"已保存特征相关性热力图: {out_path_heatmap}")
        else:
            print("特征数据不足，未生成热力图")
    except Exception as e:
        print(f"特征相关性热力图绘制失败：{e}")

if __name__ == "__main__":
    main()
