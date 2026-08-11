import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import io
import itertools
import warnings
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde, mannwhitneyu, wasserstein_distance, anderson_ksamp

# --- Configuration & Styling ---
st.set_page_config(page_title="ZetaSphere Multi-Sample Web Analyzer", layout="wide")

DEFAULT_CHANNELS = {
    '488s': ('Scatter', '#808080'),
    '405f410.lp': ('405 nm', '#800080'),
    '488f500.lp': ('488 nm', '#0000FF'),
    '520f550.lp': ('520 nm', '#008000'),
    '640f660.lp': ('640 nm', '#FF0000')
}

# --- Sidebar: Upload & Settings ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/e/eb/The_University_of_Edinburgh_logo.svg/320px-The_University_of_Edinburgh_logo.svg.png", width=150)
    st.title("ZetaSphere Analyzer")
    st.caption("Created by Daniel Gonzalez Silvera.\nImaging Facility, IRR, The University of Edinburgh.\n2026")
    
    st.header("1. Upload Data")
    uploaded_files = st.file_uploader("Drag & Drop ZetaSphere .zip files (Multiple Samples Supported)", type=["zip"], accept_multiple_files=True)
    
    st.header("2. Display Customization")
    display_style = st.radio("Plot Style", ["Both (Bar + Curve)", "Smooth Curve Only", "Histogram Only"])
    central_marker = st.radio("Draw Vertical Marker", ["None", "Mean", "Median", "Mode"])
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        bg_color = st.color_picker("Background Color", "#FFFFFF")
    with col_c2:
        axes_color = st.color_picker("Axes & Text Color", "#000000")
        
    line_width = st.slider("Line Thickness", min_value=0.5, max_value=5.0, value=1.5, step=0.5)
    
    show_grid = st.checkbox("Show Grid Lines", value=False)
    show_legend = st.checkbox("Show Legend", value=True)
    force_solid = st.checkbox("Force Solid Lines (Disable Dashes)", value=False)

    st.header("3. Statistical Analysis")
    stats_tests = st.multiselect(
        "Pairwise Comparison Tests", 
        ["Mann-Whitney U (Medians)", "Earth Mover's Distance (EMD)", "Anderson-Darling (Shape/Tails)"], 
        default=["Mann-Whitney U (Medians)"]
    )

# --- Helper Functions ---
def parse_file_info(uploaded_file):
    name_lower = uploaded_file.name.lower()
    
    measurement = "Unknown"
    if "zeta" in name_lower and "potential" in name_lower:
        measurement = "Zeta_Potential"
    elif "concentration" in name_lower:
        measurement = "Concentration"
    elif "size" in name_lower:
        measurement = "Size"
        
    channel_name = "Unknown"
    for code, (c_name, color) in DEFAULT_CHANNELS.items():
        if code.lower() in name_lower:
            channel_name = c_name
            break
            
    return measurement, channel_name

def extract_dataframe(uploaded_zip):
    try:
        with zipfile.ZipFile(uploaded_zip) as z:
            csv_files = [f for f in z.namelist() if f.endswith('measurement_result.csv')]
            if not csv_files: # Fallback if named differently
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if csv_files:
                with z.open(csv_files[0]) as f:
                    return pd.read_csv(f)
    except Exception as e:
        st.error(f"Error reading {uploaded_zip.name}: {e}")
    return None

def find_data_column(df, possible_names):
    for col in df.columns:
        if any(name.lower() == col.lower() for name in possible_names):
            return col
    for col in df.columns:
        if any(name.lower() in col.lower() for name in possible_names):
            return col
    return None

def get_mode_from_kde(series):
    data = series.dropna()
    if len(data) < 2:
        return np.nan
    kde = gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 1000)
    y_vals = kde(x_vals)
    return x_vals[np.argmax(y_vals)]

def run_stats_comparisons(data_a, data_b, tests_list):
    results = {}
    if len(data_a) == 0 or len(data_b) == 0:
        return results
        
    if "Mann-Whitney U (Medians)" in tests_list:
        _, p_val = mannwhitneyu(data_a, data_b, alternative='two-sided')
        results["MW p-value"] = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"
        
    if "Earth Mover's Distance (EMD)" in tests_list:
        emd = wasserstein_distance(data_a, data_b)
        results["EMD Score"] = f"{emd:.2f}"
        
    if "Anderson-Darling (Shape/Tails)" in tests_list:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ad_res = anderson_ksamp([data_a, data_b])
                ad_p = ad_res.pvalue
                results["AD p-value"] = "< 0.001" if ad_p <= 0.001 else ("> 0.250" if ad_p >= 0.25 else f"{ad_p:.3f}")
        except Exception:
            results["AD p-value"] = "Error"
            
    return results

def apply_custom_style(ax, title, xlabel, ylabel, xlim, bg, fg, grid, draw_legend=True):
    ax.set_facecolor(bg)
    ax.set_title(title, color=fg)
    ax.set_xlabel(xlabel, color=fg)
    ax.set_ylabel(ylabel, color=fg)
    if xlim is not None: ax.set_xlim(xlim)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values(): spine.set_color(fg)
    if grid: ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color=fg)
    
    if draw_legend:
        leg = ax.legend()
        if leg is not None:
            for text in leg.get_texts(): text.set_color(fg)
            leg.get_frame().set_facecolor(bg)
            leg.get_frame().set_edgecolor(fg)

def plot_custom_distribution(ax, data, feature_col, bins, x_vals, bin_width, color, style, line_width, label, display_style):
    x_data = data[feature_col].dropna()
    if len(x_data) == 0: return
    
    hist_label = label if display_style == "Histogram Only" else None
    curve_label = label if display_style in ["Smooth Curve Only", "Both (Bar + Curve)"] else None
    
    if display_style in ["Histogram Only", "Both (Bar + Curve)"]:
        ax.hist(x_data, bins=bins, histtype='step', color=color, linestyle=style, linewidth=line_width, label=hist_label)
        if display_style == "Both (Bar + Curve)":
            ax.hist(x_data, bins=bins, histtype='stepfilled', color=color, alpha=0.1)
            
    if display_style in ["Smooth Curve Only", "Both (Bar + Curve)"]:
        if len(x_data) > 1:
            kde = gaussian_kde(x_data)
            y_counts = kde(x_vals) * len(x_data) * bin_width
            ax.plot(x_vals, y_counts, color=color, linestyle=style, linewidth=line_width, label=curve_label)

def plot_central_marker(ax, data, color, marker_type):
    if marker_type == "Median":
        val = data.median()
        ax.axvline(val, color=color, linestyle=':', linewidth=1.5, alpha=0.8)
    elif marker_type == "Mode":
        val = get_mode_from_kde(data)
        if pd.notna(val):
            ax.axvline(val, color=color, linestyle=':', linewidth=1.5, alpha=0.8)
    elif marker_type == "Mean":
        val = data.mean()
        ax.axvline(val, color=color, linestyle=':', linewidth=1.5, alpha=0.8)

# --- Main App Layout & Data Processing ---
if not uploaded_files:
    st.info("Please upload your ZetaSphere .zip files in the sidebar to begin analysis.")
else:
    processed_data = {'Size': {}, 'Zeta_Potential': {}, 'Concentration': {}}
    
    for file in uploaded_files:
        measurement, channel = parse_file_info(file)
        df = extract_dataframe(file)
        if df is not None:
            if channel not in processed_data.get(measurement, {}):
                processed_data[measurement][channel] = []
            
            default_color = '#000000'
            for raw_code, (name, hex_code) in DEFAULT_CHANNELS.items():
                if name == channel: default_color = hex_code
                
            processed_data[measurement][channel].append({
                'filename': file.name,
                'df': df,
                'label': f"{channel} (Sample {len(processed_data[measurement][channel]) + 1})",
                'color': default_color,
                'active': True,
                'dilution': 1.0
            })

    with st.expander("🎨 Customize Individual Samples (Labels & Colors)", expanded=True):
        st.markdown("Toggle specific files on/off, rename them, and assign unique colors.")
        cols = st.columns(3)
        col_idx = 0
        for meas, channels in processed_data.items():
            for ch, items in channels.items():
                for idx, item in enumerate(items):
                    with cols[col_idx % 3]:
                        safe_name = item['filename'].replace('.', '_').replace(' ', '_') + f"_{meas}_{idx}"
                        st.markdown(f"**{meas}: {ch} (File {idx+1})**")
                        item['active'] = st.checkbox("Show Sample", value=item['active'], key=f"act_{safe_name}")
                        item['label'] = st.text_input("Label", value=item['label'], key=f"lbl_{safe_name}")
                        item['color'] = st.color_picker("Color", value=item['color'], key=f"col_{safe_name}")
                        st.write("---")
                    col_idx += 1

    tab1, tab2, tab3, tab4 = st.tabs(["Multi-Sample Size", "Zeta Potential", "Concentration", "Population Analysis (GMM)"])

    # ==========================================
    # TAB 1: SIZE OVERVIEW & MULTI-SAMPLE
    # ==========================================
    with tab1:
        st.header("Size Distribution & Channel Selection")
        if not processed_data['Size']:
            st.warning("No 'Size' measurement files detected.")
        else:
            st.subheader("Active Channels")
            available_channels = list(processed_data['Size'].keys())
            active_channels = st.multiselect("Toggle Channels On/Off:", available_channels, default=available_channels)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                size_bin_width = st.slider("Size Bin Width (nm)", 1, 50, 10, key="s_bin")
                max_x_size = st.number_input("Max X-Axis Limit (nm)", 100, 2000, 1000, key="s_max")
                
                has_replicates = any(len([i for i in items if i['active']]) > 1 for items in processed_data['Size'].values())
                rep_mode = "Treat Independent"
                if has_replicates:
                    st.info("Multiple active samples detected in the same channel.")
                    rep_mode = st.radio("Multi-Sample Handling:", ["Treat Independent", "Pool Data"], key="s_repmode")
            
            with col2:
                fig_size, ax_size = plt.subplots(figsize=(10, 5))
                fig_size.patch.set_facecolor(bg_color)
                
                line_styles = ['-', '--', ':', '-.']
                bins = np.arange(0, max_x_size + size_bin_width, size_bin_width)
                x_vals = np.linspace(0, max_x_size, 500)
                
                drawn_any = False
                for ch in active_channels:
                    active_items = [i for i in processed_data['Size'][ch] if i['active']]
                    if not active_items: continue
                    drawn_any = True
                    
                    if rep_mode == "Pool Data":
                        combined_df = pd.concat([i['df'] for i in active_items], ignore_index=True)
                        feature_col = find_data_column(combined_df, ['particle size', 'size', 'diameter'])
                        if feature_col:
                            plot_custom_distribution(ax_size, combined_df, feature_col, bins, x_vals, size_bin_width, 
                                                     active_items[0]['color'], '-', line_width, f"{ch} (Pooled)", display_style)
                            if central_marker != "None":
                                plot_central_marker(ax_size, combined_df[feature_col].dropna(), active_items[0]['color'], central_marker)
                    else:
                        for idx, item in enumerate(active_items):
                            feature_col = find_data_column(item['df'], ['particle size', 'size', 'diameter'])
                            style = '-' if force_solid else line_styles[idx % len(line_styles)]
                            if feature_col:
                                plot_custom_distribution(ax_size, item['df'], feature_col, bins, x_vals, size_bin_width, 
                                                         item['color'], style, line_width, item['label'], display_style)
                                if central_marker != "None":
                                    plot_central_marker(ax_size, item['df'][feature_col].dropna(), item['color'], central_marker)
                                
                apply_custom_style(ax_size, "Size Distribution", "Hydrodynamic Diameter (nm)", "Count", (0, max_x_size), bg_color, axes_color, show_grid, draw_legend=(drawn_any and show_legend))
                st.pyplot(fig_size)

            st.markdown("---")
            st.subheader("📊 Statistical Comparison (Size)")
            
            entities = []
            if rep_mode == "Pool Data":
                for ch in active_channels:
                    active_items = [i for i in processed_data['Size'][ch] if i['active']]
                    if not active_items: continue
                    combined_df = pd.concat([i['df'] for i in active_items], ignore_index=True)
                    f_col = find_data_column(combined_df, ['particle size', 'size', 'diameter'])
                    if f_col:
                        entities.append({"label": f"{ch} (Pooled)", "data": combined_df[f_col].dropna()})
            else:
                for ch in active_channels:
                    active_items = [i for i in processed_data['Size'][ch] if i['active']]
                    for item in active_items:
                        f_col = find_data_column(item['df'], ['particle size', 'size', 'diameter'])
                        if f_col:
                            entities.append({"label": f"[{ch}] {item['label']}", "data": item['df'][f_col].dropna()})
            
            if entities:
                summary_list = []
                for ent in entities:
                    d = ent["data"]
                    summary_list.append({
                        "Sample": ent["label"],
                        "Count": len(d),
                        "Mean (nm)": round(d.mean(), 1),
                        "Median (nm)": round(d.median(), 1),
                        "Mode (nm)": round(get_mode_from_kde(d), 1),
                        "SD (nm)": round(d.std(), 1)
                    })
                st.dataframe(pd.DataFrame(summary_list), use_container_width=True)
                
                if len(entities) > 1 and stats_tests:
                    st.info("""
                    **📚 Understanding the Statistical Tests:**
                    * **Mann-Whitney U:** Gives a p-value for the shift in the median.
                    * **Anderson-Darling:** Gives a p-value testing if the overall shape and tails of the distributions are different.
                    * **Earth Mover's Distance (EMD):** This does not yield a p-value. Instead, it gives a physical "Distance Score" (in nm or mV). A score of 0.0 means the histograms are identical. The higher the number, the more physical "work" is required to make one curve look like the other.
                    """)
                    st.markdown("*All-vs-All Pairwise Comparisons*")
                    pairwise_list = []
                    for ent_a, ent_b in itertools.combinations(entities, 2):
                        row_data = {"Sample A": ent_a["label"], "Sample B": ent_b["label"]}
                        stat_results = run_stats_comparisons(ent_a["data"], ent_b["data"], stats_tests)
                        row_data.update(stat_results)
                        pairwise_list.append(row_data)
                    st.dataframe(pd.DataFrame(pairwise_list), use_container_width=True)

    # ==========================================
    # TAB 2: ZETA POTENTIAL
    # ==========================================
    with tab2:
        st.header("Zeta Potential Histogram")
        st.info("💡 **Statistical Note:** The Mean is the standard measurement for charge distributions. **Also, this tab is designed to compare different channels from the *same* sample.** If you upload multiple independent samples, the analysis will pool them together.")

        if not processed_data['Zeta_Potential']:
            st.warning("No 'Zeta_Potential' measurement files detected.")
        else:
            available_zeta_channels = list(processed_data['Zeta_Potential'].keys())
            active_zeta_channels = st.multiselect("Toggle Zeta Channels On/Off:", available_zeta_channels, default=available_zeta_channels, key="z_toggle")
            
            col3, col4 = st.columns([1, 3])
            with col3:
                zeta_bin_width = st.slider("Zeta Bin Width (mV)", 1, 20, 5, key="z_bin")
                min_x_zeta = st.number_input("Min X-Axis Limit (mV)", -200, 0, -100, key="z_min")
                max_x_zeta = st.number_input("Max X-Axis Limit (mV)", 0, 200, 100, key="z_max")
                
                has_replicates_zeta = any(len([i for i in items if i['active']]) > 1 for items in processed_data['Zeta_Potential'].values())
                rep_mode_zeta = "Treat Independent"
                if has_replicates_zeta:
                    st.info("Multiple active samples detected in the same channel.")
                    rep_mode_zeta = st.radio("Multi-Sample Handling:", ["Treat Independent", "Pool Data"], key="z_repmode")
                
            with col4:
                fig_zeta, ax_zeta = plt.subplots(figsize=(10, 5))
                fig_zeta.patch.set_facecolor(bg_color)
                
                bins = np.arange(min_x_zeta, max_x_zeta + zeta_bin_width, zeta_bin_width)
                x_vals = np.linspace(min_x_zeta, max_x_zeta, 500)
                line_styles = ['-', '--', ':', '-.']
                
                drawn_zeta = False
                for ch in active_zeta_channels:
                    active_items = [i for i in processed_data['Zeta_Potential'][ch] if i['active']]
                    if not active_items: continue
                    drawn_zeta = True
                    
                    if rep_mode_zeta == "Pool Data":
                        combined_df = pd.concat([i['df'] for i in active_items], ignore_index=True)
                        feature_col = find_data_column(combined_df, ['zeta potential'])
                        if feature_col:
                            plot_custom_distribution(ax_zeta, combined_df, feature_col, bins, x_vals, zeta_bin_width, 
                                                     active_items[0]['color'], '-', line_width, f"{ch} (Pooled)", display_style)
                            if central_marker != "None":
                                plot_central_marker(ax_zeta, combined_df[feature_col].dropna(), active_items[0]['color'], central_marker)
                    else:
                        for idx, item in enumerate(active_items):
                            feature_col = find_data_column(item['df'], ['zeta potential'])
                            style = '-' if force_solid else line_styles[idx % len(line_styles)]
                            if feature_col:
                                plot_custom_distribution(ax_zeta, item['df'], feature_col, bins, x_vals, zeta_bin_width, 
                                                         item['color'], style, line_width, item['label'], display_style)
                                if central_marker != "None":
                                    plot_central_marker(ax_zeta, item['df'][feature_col].dropna(), item['color'], central_marker)
                
                apply_custom_style(ax_zeta, "Zeta Potential Distribution", "Zeta Potential (mV)", "Count", (min_x_zeta, max_x_zeta), bg_color, axes_color, show_grid, draw_legend=(drawn_zeta and show_legend))
                st.pyplot(fig_zeta)

            st.markdown("---")
            st.subheader("📊 Statistical Comparison (Zeta Potential)")
            
            zeta_entities = []
            if rep_mode_zeta == "Pool Data":
                for ch in active_zeta_channels:
                    active_items = [i for i in processed_data['Zeta_Potential'][ch] if i['active']]
                    if not active_items: continue
                    combined_df = pd.concat([i['df'] for i in active_items], ignore_index=True)
                    f_col = find_data_column(combined_df, ['zeta potential'])
                    if f_col:
                        zeta_entities.append({"label": f"{ch} (Pooled)", "data": combined_df[f_col].dropna()})
            else:
                for ch in active_zeta_channels:
                    active_items = [i for i in processed_data['Zeta_Potential'][ch] if i['active']]
                    for item in active_items:
                        f_col = find_data_column(item['df'], ['zeta potential'])
                        if f_col:
                            zeta_entities.append({"label": f"[{ch}] {item['label']}", "data": item['df'][f_col].dropna()})
            
            if zeta_entities:
                summary_list = []
                for ent in zeta_entities:
                    d = ent["data"]
                    summary_list.append({
                        "Sample": ent["label"],
                        "Count": len(d),
                        "Mean (mV)": round(d.mean(), 1),
                        "SD (mV)": round(d.std(), 1)
                    })
                st.dataframe(pd.DataFrame(summary_list), use_container_width=True)
                
                if len(zeta_entities) > 1 and stats_tests:
                    st.info("""
                    **📚 Understanding the Statistical Tests:**
                    * **Mann-Whitney U:** Gives a p-value for the shift in the median.
                    * **Anderson-Darling:** Gives a p-value testing if the overall shape and tails of the distributions are different.
                    * **Earth Mover's Distance (EMD):** This does not yield a p-value. Instead, it gives a physical "Distance Score" (in nm or mV). A score of 0.0 means the histograms are identical. The higher the number, the more physical "work" is required to make one curve look like the other.
                    """)
                    st.markdown("*All-vs-All Pairwise Comparisons*")
                    pairwise_list = []
                    for ent_a, ent_b in itertools.combinations(zeta_entities, 2):
                        row_data = {"Sample A": ent_a["label"], "Sample B": ent_b["label"]}
                        stat_results = run_stats_comparisons(ent_a["data"], ent_b["data"], stats_tests)
                        row_data.update(stat_results)
                        pairwise_list.append(row_data)
                    st.dataframe(pd.DataFrame(pairwise_list), use_container_width=True)

    # ==========================================
    # TAB 3: CONCENTRATION
    # ==========================================
    with tab3:
        st.header("Concentration Analysis")
        st.info("""
        💡 **Note:** Concentration is a single absolute bulk value per scan. To generate **Standard Deviation Error Bars**, simply assign the exact same 'Label' to your biological replicates in the Customize menu above! 
        
        **The dilution factor (df) can typically be found at the end of the sample's original folder name (e.g., _df250000).**
        """)
        
        if not processed_data.get('Concentration'):
            st.warning("No 'Concentration' measurement files detected.")
        else:
            available_conc_channels = list(processed_data['Concentration'].keys())
            active_conc_channels = st.multiselect("Active Concentration Channels:", available_conc_channels, default=available_conc_channels, key="c_toggle")
            
            col_c1, col_c2 = st.columns([1, 3])
            with col_c1:
                machine_vol = st.number_input("Machine Scanned Volume (mL)", value=1.0e-8, format="%.2e", help="The physical volume scanned by the laser. Used to calculate particles/mL from the raw particle count.")
                st.markdown("---")
                
                st.markdown("### 💧 Set Dilution Factors")
                for ch in active_conc_channels:
                    for item in processed_data['Concentration'][ch]:
                        if item['active']:
                            item['dilution'] = st.number_input(f"{item['label']} ({ch})", value=item.get('dilution', 1.0), format="%.1e", key=f"dil_tab3_{item['filename']}_{ch}")
                
                st.markdown("---")
                graph_type = st.radio("Graph Style", ["Bar Chart", "Dot Plot (Strip)", "Box Plot"])
                grouping = st.radio("Group By (X-Axis)", ["Channel (Compare Samples)", "Sample (Compare Channels)"])
                
            with col_c2:
                conc_data = []
                palette_dict = {}
                
                for ch in active_conc_channels:
                    active_items = [i for i in processed_data['Concentration'][ch] if i['active']]
                    for item in active_items:
                        c_val = (len(item['df']) / machine_vol) * item['dilution']
                        conc_data.append({
                            'Channel': ch,
                            'Sample Label': item['label'],
                            'Concentration (particles/mL)': c_val,
                            'Color': item['color']
                        })
                        
                        if grouping == "Sample (Compare Channels)":
                            palette_dict[ch] = item['color']
                        else:
                            palette_dict[item['label']] = item['color']
                            
                if conc_data:
                    df_conc = pd.DataFrame(conc_data)
                    
                    fig_conc, ax_conc = plt.subplots(figsize=(10, 5))
                    fig_conc.patch.set_facecolor(bg_color)
                    ax_conc.set_facecolor(bg_color)
                    
                    x_col = 'Channel' if grouping == "Channel (Compare Samples)" else 'Sample Label'
                    hue_col = 'Sample Label' if grouping == "Channel (Compare Samples)" else 'Channel'
                    
                    if graph_type == "Bar Chart":
                        sns.barplot(data=df_conc, x=x_col, y='Concentration (particles/mL)', hue=hue_col, palette=palette_dict, errorbar='sd', capsize=0.1, ax=ax_conc, edgecolor=axes_color)
                    elif graph_type == "Dot Plot (Strip)":
                        sns.stripplot(data=df_conc, x=x_col, y='Concentration (particles/mL)', hue=hue_col, palette=palette_dict, dodge=True, size=8, ax=ax_conc, edgecolor=axes_color, linewidth=1)
                    elif graph_type == "Box Plot":
                        sns.boxplot(data=df_conc, x=x_col, y='Concentration (particles/mL)', hue=hue_col, palette=palette_dict, ax=ax_conc, fliersize=5)
                    
                    ax_conc.set_title("Total Particle Concentration", color=axes_color)
                    ax_conc.set_xlabel(x_col, color=axes_color)
                    ax_conc.set_ylabel("Concentration (particles/mL)", color=axes_color)
                    ax_conc.tick_params(colors=axes_color)
                    for spine in ax_conc.spines.values(): spine.set_color(axes_color)
                    if show_grid: ax_conc.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color=axes_color)
                    
                    if show_legend:
                        leg = ax_conc.legend(title=hue_col)
                        if leg is not None:
                            for text in leg.get_texts(): text.set_color(axes_color)
                            if leg.get_title(): leg.get_title().set_color(axes_color)
                            leg.get_frame().set_facecolor(bg_color)
                            leg.get_frame().set_edgecolor(axes_color)
                    else:
                        if ax_conc.get_legend(): ax_conc.get_legend().remove()
                        
                    st.pyplot(fig_conc)

            st.markdown("---")
            st.subheader("📊 Concentration Data")
            if conc_data:
                df_summary = df_conc.groupby(['Channel', 'Sample Label']).agg(
                    Replicates=('Concentration (particles/mL)', 'count'),
                    Mean_Concentration=('Concentration (particles/mL)', 'mean'),
                    SD_Concentration=('Concentration (particles/mL)', 'std')
                ).reset_index()
                
                df_summary['Mean_Concentration'] = df_summary['Mean_Concentration'].apply(lambda x: f"{x:.2e}")
                df_summary['SD_Concentration'] = df_summary['SD_Concentration'].fillna(0).apply(lambda x: f"{x:.2e}")
                
                st.dataframe(df_summary, use_container_width=True)

    # ==========================================
    # TAB 4: POPULATION ANALYSIS (GMM)
    # ==========================================
    with tab4:
        st.header("Advanced Population Analysis")
        st.markdown("This module calculates GMM sub-populations, stats, and positivity rates for activated channels. (Active files in the same channel are pooled).")
        
        if not processed_data['Size']:
            st.warning("No 'Size' measurement files detected.")
        else:
            available_gmm_channels = list(processed_data['Size'].keys())
            active_gmm_channels = st.multiselect("Select Channels for GMM Analysis:", available_gmm_channels, default=available_gmm_channels, key="gmm_toggle")
            
            col_set1, col_set2 = st.columns([1, 3])
            with col_set1:
                max_pops = st.number_input("Max Populations to Test", 1, 6, 4)
                gmm_bin_width = st.slider("GMM Histogram Bin Width (nm)", min_value=1, max_value=50, value=10, step=1)
                gmm_x_max = st.number_input("GMM X-Axis Max", 100, 2000, 500)
                run_gmm_btn = st.button("🚀 Run Population Analysis", use_container_width=True)
            
            if run_gmm_btn:
                st.session_state['gmm_run'] = True

            if st.session_state.get('gmm_run', False) and active_gmm_channels:
                with st.spinner("Calculating Gaussian Mixture Models... Please wait."):
                    gmm_results = {}
                    master_feature_col = None
                    
                    for ch in active_gmm_channels:
                        active_items = [i for i in processed_data['Size'][ch] if i['active']]
                        if not active_items: continue
                        
                        df_clean = pd.concat([i['df'] for i in active_items], ignore_index=True)
                        base_color = active_items[0]['color']
                        filenames_used = [i['filename'] for i in active_items]
                        
                        feature_col = find_data_column(df_clean, ['particle size', 'size', 'diameter'])
                        if not feature_col: continue
                        if master_feature_col is None: master_feature_col = feature_col
                        
                        df_clean = df_clean.dropna(subset=[feature_col]).copy()
                        df_clean = df_clean[(df_clean[feature_col] > 0) & (df_clean[feature_col] <= gmm_x_max)]
                        if df_clean.empty: continue
                        
                        X_log = np.log(df_clean[[feature_col]].values)
                        bic_scores, models = [], []
                        for n in range(1, max_pops + 1):
                            gmm = GaussianMixture(n_components=n, random_state=42)
                            gmm.fit(X_log)
                            bic_scores.append(gmm.bic(X_log))
                            models.append(gmm)
                            
                        best_n = np.argmin(bic_scores) + 1
                        df_clean['Population'] = models[best_n - 1].predict(X_log) + 1
                        
                        total_particles = len(df_clean)
                        
                        summary = df_clean.groupby('Population')[feature_col].agg(Mean='mean', Median=np.median, Mode=get_mode_from_kde, SD='std', Count='count')
                        summary['Percentage (%)'] = (summary['Count'] / total_particles) * 100
                        
                        summary['Mean (nm)'] = summary['Mean'].round(1)
                        summary['Median (nm)'] = summary['Median'].round(1)
                        summary['Mode (nm)'] = summary['Mode'].round(1)
                        summary['SD (nm)'] = summary['SD'].round(1)
                        summary = summary.drop(columns=['Mean', 'Median', 'Mode', 'SD']).reset_index()
                        
                        summary.insert(0, 'Channel', ch)
                        
                        gmm_results[ch] = {
                            'df': df_clean, 'summary': summary, 'bic': bic_scores, 
                            'best_n': best_n, 'color': base_color, 'filenames': filenames_used,
                            'feature_col': feature_col
                        }
                    
                    if not gmm_results:
                        st.warning("No active valid particles found for selected channels.")
                    else:
                        global_stats_list = []
                        
                        scatter_data = None
                        if 'Scatter' in gmm_results:
                            scatter_data = gmm_results['Scatter']['df'][gmm_results['Scatter']['feature_col']].dropna()
                        
                        for ch, res in gmm_results.items():
                            f_col = res['feature_col']
                            ch_data = res['df'][f_col].dropna()
                            ch_count = len(ch_data)
                            ch_median = ch_data.median()
                            
                            pos_rate_str = "N/A"
                            stat_results = {}
                            
                            if ch != 'Scatter' and scatter_data is not None:
                                pos_rate = (ch_count / len(scatter_data)) * 100
                                pos_rate_str = f"{pos_rate:.1f}%*"
                                
                                if stats_tests:
                                    stat_results = run_stats_comparisons(scatter_data, ch_data, stats_tests)
                            
                            global_row = {
                                'Channel': ch, 
                                'Total Particle Count': ch_count, 
                                'Global Median Size (nm)': round(ch_median, 2), 
                                'Positivity Rate': pos_rate_str
                            }
                            global_row.update(stat_results)
                            global_stats_list.append(global_row)
                        
                        st.markdown("### 📊 Global Overview")
                        if len(gmm_results) > 1: st.caption("* Positivity rate is calculated by particle count ratios, not true co-localization.")
                        st.dataframe(pd.DataFrame(global_stats_list), use_container_width=True)
                        
                        st.markdown("### 🧬 Sub-Population Details")
                        for ch in gmm_results:
                            st.markdown(f"**{ch} Channel Breakdown:**")
                            st.dataframe(gmm_results[ch]['summary'], use_container_width=True)
                        
                        min_val_ext = 0
                        max_val_ext = gmm_x_max
                        x_vals_ext = np.linspace(min_val_ext, max_val_ext, 500)
                        master_curves = pd.DataFrame({'Hydrodynamic Diameter (nm)': x_vals_ext})
                        
                        bins_ext = np.arange(min_val_ext, max_val_ext + gmm_bin_width, gmm_bin_width)
                        master_hists = pd.DataFrame({'Bin_Start (nm)': bins_ext[:-1], 'Bin_End (nm)': bins_ext[1:]})
                        
                        for ch, res in gmm_results.items():
                            df_ch = res['df']
                            f_col = res['feature_col']
                            best_n = res['best_n']
                            
                            if len(df_ch) > 1:
                                kde_total = gaussian_kde(df_ch[f_col])
                                master_curves[f'{ch}_Total_Density'] = kde_total(x_vals_ext) * len(df_ch)
                                
                            counts_total, _ = np.histogram(df_ch[f_col], bins=bins_ext)
                            master_hists[f'{ch}_Total_Count'] = counts_total
                            
                            for pop in range(1, best_n + 1):
                                pop_data = df_ch[df_ch['Population'] == pop][f_col]
                                if len(pop_data) > 1:
                                    kde_pop = gaussian_kde(pop_data)
                                    master_curves[f'{ch}_Pop_{pop}_Density'] = kde_pop(x_vals_ext) * len(pop_data)
                                counts_pop, _ = np.histogram(pop_data, bins=bins_ext)
                                master_hists[f'{ch}_Pop_{pop}_Count'] = counts_pop

                        first_ch = list(gmm_results.keys())[0]
                        first_filename = gmm_results[first_ch]['filenames'][0]
                        base_name_export = first_filename.replace('.zip', '')
                        dynamic_excel_name = f"{base_name_export}_Population_Analysis.xlsx"

                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            pd.concat([pd.DataFrame(global_stats_list), pd.DataFrame({'Channel': ['*Positivity result not based on colocalization analysis']})], ignore_index=True).to_excel(writer, sheet_name='1_Global_Stats', index=False)
                            pd.concat([res['summary'] for res in gmm_results.values()], ignore_index=True).to_excel(writer, sheet_name='2_Population_Summary', index=False)
                            master_bic = pd.DataFrame({'Number of Populations Tested': range(1, max_pops + 1)})
                            for ch, res in gmm_results.items(): master_bic[f'{ch} BIC Score'] = res['bic']
                            master_bic.to_excel(writer, sheet_name='3_BIC_Scores', index=False)
                            master_curves.to_excel(writer, sheet_name='4_Graph_Curves', index=False)
                            master_hists.to_excel(writer, sheet_name='5_Histogram_Data', index=False)

                        num_channels = len(gmm_results)
                        total_rows = 1 + num_channels if num_channels > 1 else 1
                        fig_gmm = plt.figure(figsize=(14, 5 * total_rows))
                        fig_gmm.patch.set_facecolor(bg_color)
                        bins = np.arange(0, gmm_x_max + gmm_bin_width, gmm_bin_width)
                        x_vals = np.linspace(0, gmm_x_max, 500)

                        if num_channels == 1:
                            ax1, ax2 = plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)
                            ch, res = list(gmm_results.items())[0]
                            ax1.plot(range(1, max_pops + 1), res['bic'], marker='o', linestyle='-', color=res['color'])
                            ax1.set_xticks(range(1, max_pops + 1))  
                            apply_custom_style(ax1, 'Model Scoring (Lowest BIC Wins)', 'Populations Tested', 'BIC Score', None, bg_color, axes_color, show_grid, draw_legend=False)
                            
                            f_col = res['feature_col']
                            sns.histplot(data=res['df'], x=f_col, hue='Population', palette='viridis', element='step' if display_style != "Smooth Curve Only" else None, binwidth=gmm_bin_width, kde=True, fill=display_style != "Smooth Curve Only", alpha=0.2 if display_style != "Smooth Curve Only" else 0, line_kws={'linewidth': line_width}, ax=ax2)
                            apply_custom_style(ax2, f"{ch} Particles Grouped into {res['best_n']} Populations", "Hydrodynamic Diameter (nm)", "Count", (0, gmm_x_max), bg_color, axes_color, show_grid, draw_legend=show_legend)
                        
                        else:
                            ax1, ax2 = plt.subplot(total_rows, 2, 1), plt.subplot(total_rows, 2, 2)
                            for ch, res in gmm_results.items():
                                c = res['color']
                                ax1.plot(range(1, max_pops + 1), res['bic'], marker='o', linestyle='-', color=c, label=ch)
                                plot_custom_distribution(ax2, res['df'], res['feature_col'], bins, x_vals, gmm_bin_width, c, '-', line_width, ch, display_style)
                            
                            ax1.set_xticks(range(1, max_pops + 1)) 
                            apply_custom_style(ax1, 'Model Scoring (Lowest BIC Wins)', 'Populations Tested', 'BIC Score', None, bg_color, axes_color, show_grid, draw_legend=show_legend)
                            apply_custom_style(ax2, "Multi-Channel Size Distribution Overlay", "Hydrodynamic Diameter (nm)", "Count", (0, gmm_x_max), bg_color, axes_color, show_grid, draw_legend=show_legend)
                            
                            for i, (ch, res) in enumerate(gmm_results.items()):
                                ax_sub = plt.subplot(total_rows, 1, i + 2)
                                f_col = res['feature_col']
                                sns.histplot(data=res['df'], x=f_col, hue='Population', palette='viridis', element='step' if display_style != "Smooth Curve Only" else None, binwidth=gmm_bin_width, kde=True, fill=display_style != "Smooth Curve Only", alpha=0.2 if display_style != "Smooth Curve Only" else 0, line_kws={'linewidth': line_width}, ax=ax_sub)
                                apply_custom_style(ax_sub, f"Sub-population Breakdown: {ch} ({res['best_n']} Populations Found)", "Hydrodynamic Diameter (nm)", "Count", (0, gmm_x_max), bg_color, axes_color, show_grid, draw_legend=show_legend)

                        plt.tight_layout()
                        st.pyplot(fig_gmm)
                        
                        st.download_button(
                            label="📥 Download Excel Statistics Data",
                            data=excel_buffer.getvalue(),
                            file_name=dynamic_excel_name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
