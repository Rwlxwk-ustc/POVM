import streamlit as st
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import binom, norm, poisson
import cvxpy as cp
import matplotlib.patches as mpatches
import warnings
import math

# ==========================================
# 0. 页面配置与字体处理
# ==========================================
st.set_page_config(page_title="POVM & FI 交互计算平台", layout="wide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

# 兼容 Mac/Windows 字体（注意：Streamlit Cloud 环境下若无此字体可能会显示方块）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 物理计算核心函数 (添加缓存机制提升性能)
# ==========================================
@st.cache_data(show_spinner=False)
def build_loss_matrix(M, efficiency):
    L = np.zeros((M + 1, M + 1), dtype=np.float64)
    for n in range(M + 1):
        for m in range(n + 1):
            L[m, n] = binom.pmf(m, n, efficiency)
    return L

@st.cache_data(show_spinner=False)
def build_pc_response_matrix(N, M):
    S = np.zeros((M + 1, N + 1), dtype=np.float64)
    S[0, 0] = 1.0
    for m in range(1, M + 1):
        for k in range(1, min(m, N) + 1):
            S[m, k] = k * S[m-1, k] + S[m-1, k-1]
    C = np.zeros((N + 1, M + 1), dtype=np.float64)
    C[0, 0] = 1.0
    for m in range(1, M + 1):
        for k in range(1, min(m, N) + 1):
            permutations = math.perm(N, k)
            C[k, m] = (permutations * S[m, k]) / (N ** m)
    return C

@st.cache_data(show_spinner=False)
def build_pd_response_matrix(bins, M, sigma):
    N_bins = len(bins) - 1
    C = np.zeros((N_bins, M + 1), dtype=np.float64)
    for m in range(M + 1):
        for j in range(N_bins):
            p_high = norm.cdf(bins[j+1], loc=m, scale=sigma)
            p_low = norm.cdf(bins[j], loc=m, scale=sigma)
            C[j, m] = p_high - p_low
    return C

def generate_experiment_data(I_arr, theta_mat, k_arr, N_trials=2000):
    N_out, M_len = theta_mat.shape
    N_probes = len(I_arr)
    F = np.zeros((N_probes, M_len))
    for i, I in enumerate(I_arr):
        F[i, :] = poisson.pmf(k_arr, I)
    P_noisy = np.zeros((N_probes, N_out))
    P_clean = F @ theta_mat.T  
    for i in range(N_probes):
        true_p = P_clean[i, :]
        true_p = true_p / np.sum(true_p)
        counts = np.random.multinomial(N_trials, true_p)
        P_noisy[i, :] = counts / N_trials
    return F, P_noisy

def solve_povm(F_mat, P_exp, N_out, M_len, gamma=0.5):
    Theta_var = cp.Variable((N_out, M_len))
    data_fidelity = cp.sum_squares(P_exp - F_mat @ Theta_var.T)
    smoothness = cp.sum_squares(Theta_var[:, 2:] - 2 * Theta_var[:, 1:-1] + Theta_var[:, :-2])
    objective = cp.Minimize(data_fidelity + gamma * smoothness)
    constraints = [Theta_var >= 0, cp.sum(Theta_var, axis=0) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)  
    Theta_res = np.clip(Theta_var.value, 0, 1)
    Theta_res = Theta_res / np.sum(Theta_res, axis=0, keepdims=True)
    return Theta_res

# ==========================================
# 2. UI 侧边栏构建
# ==========================================
st.sidebar.markdown("### ⚙️ 参数控制")
st.sidebar.info("💡 提示：你可以直接点击滑块右侧的数字，手动输入精确数值。")

with st.sidebar.form("param_form"):
    eta = st.slider("效率 (η)", min_value=0.1, max_value=1.0, value=0.75, step=0.05)
    sigma_el = st.slider("噪声 (σ_el)", min_value=0.1, max_value=3.0, value=1.2, step=0.1)
    M_max = st.slider("截断 (M)", min_value=50, max_value=150, value=110, step=1)
    
    st.divider()
    alpha_0 = st.slider("探针起点", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    alpha_max = st.slider("探针最大", min_value=5.0, max_value=15.0, value=9.0, step=0.5)
    N_alpha = st.slider("探针数量", min_value=5, max_value=30, value=16, step=1)
    
    st.divider()
    N_pc = st.slider("PC 通道", min_value=2, max_value=32, value=12, step=1)
    N_pd = st.slider("PD 划分", min_value=2, max_value=32, value=18, step=1)
    
    submit_btn = st.form_submit_button("🚀 开始计算", use_container_width=True)

# 渲染全局图例 (独立于计算之外，常驻侧边栏下方)
st.sidebar.divider()
st.sidebar.markdown("### 📖 全局图例")

c_pc = plt.cm.viridis(np.linspace(0, 0.9, N_pc))
c_pd = plt.cm.plasma(np.linspace(0, 0.9, N_pd))
c_dir_pc, c_dir_pd, c_bhd_pc, c_bhd_pd = '#FF8C00', '#DC143C', '#00BFFF', '#0000CD'
bg_dir, bg_bhd = '#E8F5E9', '#F3E5F5'

legend_fig = plt.figure(figsize=(3, 4.5), dpi=100)
legend_fig.patch.set_facecolor('#F8F9FA') 
gs_leg = legend_fig.add_gridspec(3, 1, height_ratios=[0.15, 0.15, 1], hspace=0.6)
ax_cb1 = legend_fig.add_subplot(gs_leg[0])
ax_cb2 = legend_fig.add_subplot(gs_leg[1])
ax_fi_leg = legend_fig.add_subplot(gs_leg[2])

cmap_pc = mpl.colors.ListedColormap(c_pc)
norm_pc = mpl.colors.Normalize(vmin=-0.5, vmax=N_pc-0.5)
cb1 = mpl.colorbar.ColorbarBase(ax_cb1, cmap=cmap_pc, norm=norm_pc, orientation='horizontal')
cb1.ax.set_title("Row 1 - CH PC", fontsize=9, fontweight='bold')
cb1.set_ticks([0, N_pc//2, N_pc-1])
cb1.ax.tick_params(labelsize=8)

cmap_pd = mpl.colors.ListedColormap(c_pd)
norm_pd = mpl.colors.Normalize(vmin=-0.5, vmax=N_pd-0.5)
cb2 = mpl.colorbar.ColorbarBase(ax_cb2, cmap=cmap_pd, norm=norm_pd, orientation='horizontal')
cb2.ax.set_title("Row 1 - CH PD", fontsize=9, fontweight='bold')
cb2.set_ticks([0, N_pd//2, N_pd-1])
cb2.ax.tick_params(labelsize=8)

ax_fi_leg.axis('off')
fi_handles = [
    Line2D([0], [0], color='k', ls='-.', lw=1.5, label='QFI Bound'),
    Line2D([0], [0], color=c_dir_pc, lw=1.5, label='Direct PC'),
    Line2D([0], [0], color=c_dir_pd, lw=1.5, label='Direct PD'),
    Line2D([0], [0], color=c_bhd_pc, lw=1.5, label='BHD PC'),
    Line2D([0], [0], color=c_bhd_pd, lw=1.5, label='BHD PD'),
    mpatches.Patch(facecolor=bg_dir, edgecolor='gray', label='Opt Dir'),
    mpatches.Patch(facecolor=bg_bhd, edgecolor='gray', label='Opt BHD'),
    mpatches.Patch(facecolor='white', edgecolor='gray', hatch='//', label='Opt PC (//)'),
    mpatches.Patch(facecolor='white', edgecolor='gray', hatch='\\\\', label='Opt PD (\\\\)')
]
ax_fi_leg.legend(handles=fi_handles, loc='upper center', ncol=1, fontsize=8, frameon=False)
st.sidebar.pyplot(legend_fig)
plt.close(legend_fig)

# ==========================================
# 3. 核心计算与主图表渲染
# ==========================================
st.title("量子探测器 POVM 层析与 FI 计算仪表盘")

if not submit_btn:
    st.info("👈 请在左侧侧边栏调整参数，然后点击 **【🚀 开始计算】** 生成数据和图表。")
else:
    with st.spinner("🔄 正在进行矩阵推导与凸优化求解...这可能需要几秒钟..."):
        # --- 准备数据 ---
        N_modes = N_pc - 1
        k_arr = np.arange(M_max + 1)
        max_expected_intensity = alpha_max**2  
        v_max = eta * max_expected_intensity + 4 * sigma_el  
        internal_edges = np.linspace(-2, v_max, N_pd - 1)
        voltage_bins = np.concatenate(([-np.inf], internal_edges, [np.inf]))
        alpha_probes = np.linspace(alpha_0, alpha_max, N_alpha)
        I_probes = alpha_probes**2

        # 理论矩阵
        L_mat = build_loss_matrix(M_max, eta)
        C_pc = build_pc_response_matrix(N_modes, M_max)
        C_pd = build_pd_response_matrix(voltage_bins, M_max, sigma_el)

        theta_pc_true = np.clip(C_pc @ L_mat, 0, 1)
        theta_pc_true /= np.sum(theta_pc_true, axis=0, keepdims=True)
        theta_pd_true = np.clip(C_pd @ L_mat, 0, 1)
        theta_pd_true /= np.sum(theta_pd_true, axis=0, keepdims=True)

        np.random.seed(42)
        F_exp, P_exp_pc = generate_experiment_data(I_probes, theta_pc_true, k_arr, 2000)
        _, P_exp_pd = generate_experiment_data(I_probes, theta_pd_true, k_arr, 2000)

        # CVXPY POVM 重构
        theta_pc_rec = solve_povm(F_exp, P_exp_pc, N_pc, M_max + 1)
        theta_pd_rec = solve_povm(F_exp, P_exp_pd, N_pd, M_max + 1)

        # 连续曲线预测
        I_continuous = np.linspace(0, np.max(I_probes) * 1.05, 200)
        F_cont = np.zeros((len(I_continuous), M_max + 1))
        for i, I in enumerate(I_continuous):
            F_cont[i, :] = poisson.pmf(k_arr, I)
        P_pred_pc = F_cont @ theta_pc_rec.T
        P_pred_pd = F_cont @ theta_pd_rec.T

        # FI 计算 (直接探测)
        theta_angles = np.linspace(0.1, np.pi/2 - 0.1, 200)
        I_theta = max_expected_intensity * np.sin(theta_angles)**2
        dI_dtheta = 2 * max_expected_intensity * np.sin(theta_angles) * np.cos(theta_angles)

        var_pd = eta * I_theta + sigma_el**2
        F_pc_theo = (eta / I_theta) * (dI_dtheta**2)
        F_pd_theo = (eta**2 * dI_dtheta**2) / var_pd + (eta**2 * dI_dtheta**2) / (2 * var_pd**2)

        numerical_safe_threshold = 1e-12 
        def calc_dir_fi(theta_rec):
            F_out = np.zeros_like(theta_angles)
            for idx, (I, dI) in enumerate(zip(I_theta, dI_dtheta)):
                p_k = poisson.pmf(k_arr, I)
                dpk_dI = np.zeros_like(p_k)
                dpk_dI[0] = -p_k[0]; dpk_dI[1:] = p_k[:-1] - p_k[1:]
                dp_out_dtheta = theta_rec @ (dpk_dI * dI)
                p_out = theta_rec @ p_k
                valid = p_out > numerical_safe_threshold
                F_out[idx] = np.sum((dp_out_dtheta[valid]**2) / p_out[valid])
            return F_out

        F_pc_exp = calc_dir_fi(theta_pc_rec)
        F_pd_exp = calc_dir_fi(theta_pd_rec)

        # FI 计算 (BHD探测)
        beta_val = 6.0
        alpha_val = np.sqrt(max_expected_intensity)
        QFI_bound = 4 * (alpha_val**2) * np.ones_like(theta_angles)

        I_c = 0.5 * (alpha_val**2 + beta_val**2 + 2 * alpha_val * beta_val * np.cos(theta_angles))
        I_d = 0.5 * (alpha_val**2 + beta_val**2 - 2 * alpha_val * beta_val * np.cos(theta_angles))
        dIc_dtheta = -alpha_val * beta_val * np.sin(theta_angles)
        dId_dtheta = alpha_val * beta_val * np.sin(theta_angles)

        F_BHD_pc_theo = eta * (alpha_val**2) * (beta_val**2) * (np.sin(theta_angles)**2) * (1/I_c + 1/I_d)
        F_c_pd_theo = (eta**2 * dIc_dtheta**2) / (eta * I_c + sigma_el**2) + (eta**2 * dIc_dtheta**2) / (2 * (eta * I_c + sigma_el**2)**2)
        F_d_pd_theo = (eta**2 * dId_dtheta**2) / (eta * I_d + sigma_el**2) + (eta**2 * dId_dtheta**2) / (2 * (eta * I_d + sigma_el**2)**2)
        F_BHD_pd_theo = F_c_pd_theo + F_d_pd_theo

        def calc_arm_fi(I_arr, dI_arr, theta_rec):
            F_arm = np.zeros_like(I_arr)
            for idx, (I, dI) in enumerate(zip(I_arr, dI_arr)):
                p_k = poisson.pmf(k_arr, I)
                dpk_dI = np.zeros_like(p_k)
                dpk_dI[0] = -p_k[0]; dpk_dI[1:] = p_k[:-1] - p_k[1:]
                p_out = theta_rec @ p_k
                dp_out_dtheta = theta_rec @ (dpk_dI * dI)
                valid = p_out > numerical_safe_threshold
                F_arm[idx] = np.sum((dp_out_dtheta[valid]**2) / p_out[valid])
            return F_arm

        F_BHD_pc_exp = calc_arm_fi(I_c, dIc_dtheta, theta_pc_rec) + calc_arm_fi(I_d, dId_dtheta, theta_pc_rec)
        F_BHD_pd_exp = calc_arm_fi(I_c, dIc_dtheta, theta_pd_rec) + calc_arm_fi(I_d, dId_dtheta, theta_pd_rec)

        eff_dir_pc = F_pc_exp / F_pc_theo * 100
        eff_dir_pd = F_pd_exp / F_pd_theo * 100
        eff_bhd_pc = F_BHD_pc_exp / F_BHD_pc_theo * 100
        eff_bhd_pd = F_BHD_pd_exp / F_BHD_pd_theo * 100

        # ==========================================
        # 4. 图表排版与渲染
        # ==========================================
        fig = plt.figure(figsize=(14, 11), dpi=100)
        gs_main = fig.add_gridspec(3, 1, height_ratios=[1, 1.2, 1.6], hspace=0.35)
        
        # 行 1
        gs_row1 = gs_main[0].subgridspec(1, 2, wspace=0.15)
        ax1 = fig.add_subplot(gs_row1[0])
        ax2 = fig.add_subplot(gs_row1[1])
        
        # 行 2
        gs_row2 = gs_main[1].subgridspec(1, 4, wspace=0.45)
        ax3 = fig.add_subplot(gs_row2[0])
        ax4 = fig.add_subplot(gs_row2[1], projection='3d')
        ax5 = fig.add_subplot(gs_row2[2])
        ax6 = fig.add_subplot(gs_row2[3], projection='3d')

        # 行 3
        gs_row3 = gs_main[2].subgridspec(1, 2, wspace=0.15)
        ax7 = fig.add_subplot(gs_row3[0])
        ax8 = fig.add_subplot(gs_row3[1])

        # -- 渲染行 1 --
        for n in range(N_pc):
            ax1.scatter(I_probes, P_exp_pc[:, n], color=c_pc[n], s=10, alpha=0.7)
            ax1.plot(I_continuous, P_pred_pc[:, n], color=c_pc[n], lw=1.5)
        ax1.set_title("PC: Fit vs Scatter", fontsize=12, fontweight='bold')
        ax1.tick_params(labelsize=9)
        ax1.grid(True, linestyle=':', alpha=0.5)

        for j in range(N_pd):
            ax2.scatter(I_probes, P_exp_pd[:, j], color=c_pd[j], marker='s', s=10, alpha=0.7)
            ax2.plot(I_continuous, P_pred_pd[:, j], color=c_pd[j], lw=1.5)
        ax2.set_title("PD: Fit vs Scatter", fontsize=12, fontweight='bold')
        ax2.tick_params(labelsize=9)
        ax2.grid(True, linestyle=':', alpha=0.5)

        # -- 渲染行 2 --
        im1 = ax3.imshow(theta_pc_rec, aspect='auto', origin='lower', cmap='viridis', extent=[-0.5, M_max+0.5, -0.5, N_pc-0.5])
        ax3.set_title("PC POVM (2D)", fontsize=10)
        ax3.tick_params(labelsize=9)
        fig.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04).ax.tick_params(labelsize=9)

        x, y, z, dz, cols = [], [], [], [], []
        for n in range(N_pc):
            for k in range(M_max + 1):
                if theta_pc_rec[n, k] > 1e-3:
                    x.append(k)
                    y.append(n)
                    z.append(0)
                    dz.append(theta_pc_rec[n, k])
                    cols.append(c_pc[n])
        ax4.bar3d(x, y, z, 0.8, 0.8, dz, color=cols, alpha=0.8)
        ax4.set_title("PC POVM (3D)", fontsize=10)
        ax4.tick_params(labelsize=7)
        ax4.view_init(elev=25, azim=-60)

        im2 = ax5.imshow(theta_pd_rec, aspect='auto', origin='lower', cmap='plasma', extent=[-0.5, M_max+0.5, -0.5, N_pd-0.5])
        ax5.set_title("PD POVM (2D)", fontsize=10)
        ax5.tick_params(labelsize=9)
        fig.colorbar(im2, ax=ax5, fraction=0.046, pad=0.04).ax.tick_params(labelsize=9)

        x, y, z, dz, cols = [], [], [], [], []
        for j in range(N_pd):
            for k in range(M_max + 1):
                if theta_pd_rec[j, k] > 1e-3:
                    x.append(k)
                    y.append(j)
                    z.append(0)
                    dz.append(theta_pd_rec[j, k])
                    cols.append(c_pd[j])
        ax6.bar3d(x, y, z, 0.8, 0.8, dz, color=cols, alpha=0.8)
        ax6.set_title("PD POVM (3D)", fontsize=10)
        ax6.tick_params(labelsize=7)
        ax6.view_init(elev=25, azim=-60)

        # -- 渲染行 3 --
        all_curves = np.vstack([F_pc_exp, F_pd_exp, F_BHD_pc_exp, F_BHD_pd_exp])
        winner_idx = np.argmax(all_curves, axis=0)
        g_max = np.max(all_curves, axis=0)
        y_max = max(np.max(g_max)*1.15, QFI_bound[0]*1.15)
        
        ax7.fill_between(theta_angles, 0, y_max, where=(winner_idx<2), facecolor=bg_dir, alpha=0.8)
        ax7.fill_between(theta_angles, 0, y_max, where=(winner_idx>=2), facecolor=bg_bhd, alpha=0.8)
        
        ax7.fill_between(theta_angles, 0, g_max, where=(winner_idx==0), facecolor='none', edgecolor=c_dir_pc, hatch='//', alpha=0.7)
        ax7.fill_between(theta_angles, 0, g_max, where=(winner_idx==1), facecolor='none', edgecolor=c_dir_pd, hatch='\\\\', alpha=0.7)
        ax7.fill_between(theta_angles, 0, g_max, where=(winner_idx==2), facecolor='none', edgecolor=c_bhd_pc, hatch='//', alpha=0.7)
        ax7.fill_between(theta_angles, 0, g_max, where=(winner_idx==3), facecolor='none', edgecolor=c_bhd_pd, hatch='\\\\', alpha=0.7)

        ax7.plot(theta_angles, QFI_bound, 'k-.', lw=2)
        ax7.plot(theta_angles, F_pc_exp, color=c_dir_pc, lw=2)
        ax7.plot(theta_angles, F_pd_exp, color=c_dir_pd, lw=2)
        ax7.plot(theta_angles, F_BHD_pc_exp, color=c_bhd_pc, lw=2)
        ax7.plot(theta_angles, F_BHD_pd_exp, color=c_bhd_pd, lw=2)

        ax7.set_title("Global Optimization: Scheme & Detector FI", fontsize=12, fontweight='bold')
        ax7.set_ylim(0, y_max); ax7.set_xlim(theta_angles[0], theta_angles[-1])
        ax7.tick_params(labelsize=10)
        ax7.grid(True, linestyle=':', alpha=0.5)

        ax8.plot(theta_angles, eff_dir_pc, color=c_dir_pc, lw=2)
        ax8.plot(theta_angles, eff_dir_pd, color=c_dir_pd, lw=2)
        ax8.plot(theta_angles, eff_bhd_pc, color=c_bhd_pc, lw=2)
        ax8.plot(theta_angles, eff_bhd_pd, color=c_bhd_pd, lw=2)
        ax8.axhline(100, color='k', ls='--')
        
        ax8.set_title("Extraction Efficiency (%)", fontsize=12, fontweight='bold')
        ax8.set_ylim(0, max(110, np.max([eff_dir_pc, eff_dir_pd, eff_bhd_pc, eff_bhd_pd])*1.1))
        ax8.set_xlim(theta_angles[0], theta_angles[-1])
        ax8.tick_params(labelsize=10)
        ax8.grid(True, linestyle=':', alpha=0.5)

        st.pyplot(fig)
        plt.close(fig)
        st.success("✅ 计算及渲染完成！")
