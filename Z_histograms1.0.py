import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import io
import itertools
import warnings
import math
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde, mannwhitneyu, wasserstein_distance, anderson_ksamp

# --- Configuration & Styling ---
st.set_page_config(page_title="ZetaSphere Multi-Sample Web Analyzer", layout="wide")

DEFAULT_CHANNELS = {
    '488s': ('Scatter', '#808080'),
    '405': ('405 nm', '#4169E1'), # Royal Blue emission
    '488': ('488 nm', '#228B22'), # Forest Green emission
    '520': ('520 nm', '#FF4500'), # Orange-Red emission
    '640': ('640 nm', '#C71585')  # Pinkish/Medium-Violet-Red emission
}

PALETTES = {
    "Custom (Sample Colors)": None,
    "Viridis": "viridis",
    "Plasma": "plasma",
    "Pastel": "pastel",
    "Magma": "magma",
    "Coolwarm": "coolwarm",
    "Colorblind": "colorblind",
    "Inferno": "inferno",
    "Cividis": "cividis",
    "Rocket": "rocket",
    "Cubehelix": "cubehelix",
    "Tab10": "tab10",
    "Spectral": "Spectral",
    "Set1": "Set1",
    "Set2": "Set2",
    "Husl": "husl"
}

# --- Sidebar: Upload & Settings ---
with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.title("ZetaSphere Histogram Analyzer")
    st.caption("Created by Daniel Gonzalez Silvera. \nImaging Facility, IRR, The University of Edinburgh.\n2026")
    
    st.header("1. Upload Data")
    uploaded_files = st.file_uploader("Drag & Drop ZetaSphere .zip files (Multiple Samples Supported)", type=["zip"], accept_multiple_files=True)
    
    st.header("2. Display Customization")
    palette_choice = st.selectbox("Color Palette", list(PALETTES.keys()), help="Select a predefined colorset, or use your custom manual colors.")
    multi_layout = st.radio("Multi-Sample Layout", ["Overlay (Default)", "Facet Grid", "Ridgeline (Joyplot)", "Violin Plot"], help="Choose how multiple active samples are displayed.")
    
    if multi_layout == "Facet Grid":
        facet_share_y = st.checkbox("Share Y-Axis across Facets", value=True, help="Keep all subplots on the exact same vertical scale.")
    else:
        facet_share_y = False
        
    display_style = st.radio("Plot Style", ["Both (Bar + Curve)", "Smooth Curve Only", "Histogram Only"])
    central_marker = st.radio("Draw Vertical Marker", ["None", "Mean", "Median", "Mode"])
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        bg_color = st.color_picker("Background Color", "#FFFFFF")
    with col_c2:
        axes_color = st.color_picker("Axes & Text Color", "#000000")
        
    st.markdown("---")
    st.markdown("**Graph Sizing & Typography**")
    line_width = st.slider("Plot Line Thickness", min_value=0.5, max_value=5.0, value=1.5, step=0.5)
    axes_width = st.slider("Axis Box Thickness", min_value=0.5, max_value=3.0, value=1.0, step=0.5)
    title_size = st.slider("Title Font Size", min_value=8, max_value=24, value=14, step=1)
    label_size = st.slider("Axis Label & Tick Font Size", min_value=8, max_value=20, value=10, step=1)
    legend_size = st.slider("Legend Font Size", min_value=6, max_value=16, value=10, step=1)
    
    show_grid = st.checkbox("Show Grid Lines", value=False)
    show_legend = st.checkbox("Show Legend", value=True)
    # --- NEW: Conditional Legend Position ---
    if show_legend:
        legend_position = st.selectbox(
            "Legend Position", 
            ["best", "upper right", "upper left", "lower right", "lower left", "center right", "center left", "upper center", "lower center"],
            index=0
        )
    else:
        # Provide a safe invisible default so Python doesn't crash looking for the variable
        legend_position = "best"

    force_solid = st.checkbox("Force Solid Lines (Disable Dashes)", value=False)

    st.markdown("---")
    st.subheader("Advanced Data Filtering")
    expert_filtering = st.checkbox("Filtering data - experts only")
        
    if expert_filtering:
        min_trace_length = st.slider(
            "Minimum Trace Length (Frames)", 
            min_value=1, max_value=50, value=1, step=1, 
            help="Filter out particles tracked for too few frames."
        )
        st.info("💡 **Note:** The manufacturer recommended link radius is 10 pixels.")
        link_radius = st.slider(
            "Link Radius (Pixels)", 
            min_value=1.0, max_value=30.0, value=10.0, step=1.0, 
            help="The spatial tolerance used to match moving particles between the two consecutive laser recordings."
        )
    else:
        # Safe defaults when hidden
        min_trace_length = 1
        link_radius = 10.0

# --- Helper Functions ---
def parse_file_info(uploaded_file):
    filename = uploaded_file.name
    parts = filename.split('_')
    
    measurement = "Unknown"
    channel_str = ""
    
    # Extract measurement and channel string strictly by their position in the filename structure
    if len(parts) >= 4:
        meas_part = parts[2].lower()
        if meas_part == "size":
            measurement = "Size"
            channel_str = parts[3].lower()
        elif meas_part == "concentration":
            measurement = "Concentration"
            channel_str = parts[3].lower()
        elif meas_part == "colocalization":
            measurement = "Colocalization"
            channel_str = parts[3].lower()
        elif meas_part == "zeta":
            if parts[3].lower() == "potential" and len(parts) >= 5:
                measurement = "Zeta_Potential"
                channel_str = parts[4].lower()
            else:
                measurement = "Zeta_Potential"
                channel_str = parts[3].lower()
                
    # Extract clean channel names
    channel_name = "Unknown"
    if measurement == "Colocalization":
        found = []
        for code, (c_name, color) in DEFAULT_CHANNELS.items():
            idx = channel_str.find(code.lower())
            if idx != -1:
                found.append((idx, c_name, color))
        found.sort()
        if len(found) >= 2:
            channel_name = f"{found[0][1]} vs {found[1][1]}"
        else:
            channel_name = "Colocalization"
    else:
        for code, (c_name, color) in DEFAULT_CHANNELS.items():
            if code.lower() in channel_str:
                channel_name = c_name
                break
                
    return measurement, channel_name

def extract_dataframe(uploaded_zip):
    try:
        with zipfile.ZipFile(uploaded_zip) as z:
            csv_files = [f for f in z.namelist() if f.endswith('measurement_result.csv')]
            if not csv_files: csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if csv_files:
                with z.open(csv_files[0]) as f: return pd.read_csv(f)
    except Exception as e:
        st.error(f"Error reading {uploaded_zip.name}: {e}")
    return None

def find_data_column(df, possible_names):
    for col in df.columns:
        if any(name.lower() == col.lower() for name in possible_names): return col
    for col in df.columns:
        if any(name.lower() in col.lower() for name in possible_names): return col
    return None

def get_mode_from_kde(series):
    data = series.dropna()
    if len(data) < 2: return np.nan
    kde = gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 1000)
    return x_vals[np.argmax(kde(x_vals))]

def get_span(series):
    data = series.dropna()
    if len(data) < 2: return np.nan
    d10, d50, d90 = np.percentile(data, [10, 50, 90])
    return (d90 - d10) / d50 if d50 > 0 else np.nan

def get_fwhm_from_kde(series):
    data = series.dropna()
    if len(data) < 2 or data.std() == 0: return np.nan
    kde = gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 1000)
    y_vals = kde(x_vals)
    max_idx = np.argmax(y_vals)
    half_max = np.max(y_vals) / 2.0
    
    left_half = x_vals[:max_idx][np.argmin(np.abs(y_vals[:max_idx] - half_max))] if max_idx > 0 else x_vals[0]
    right_half = x_vals[max_idx:][np.argmin(np.abs(y_vals[max_idx:] - half_max))] if max_idx < len(x_vals)-1 else x_vals[-1]
    return right_half - left_half

def get_mad(series):
    data = series.dropna()
    if len(data) < 2: return np.nan
    med = np.median(data)
    return np.median(np.abs(data - med))

def run_stats_comparisons(data_a, data_b):
    results = {}
    if len(data_a) == 0 or len(data_b) == 0: return results
    
    # 1. Mann-Whitney U
    _, p_val = mannwhitneyu(data_a, data_b, alternative='two-sided')
    results["MW p-value"] = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"
    
    # 2. Earth Mover's Distance (EMD)
    results["EMD Score"] = f"{wasserstein_distance(data_a, data_b):.2f}"
    
    # 3. Anderson-Darling
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ad_p = anderson_ksamp([data_a, data_b]).pvalue
            results["AD p-value"] = "< 0.001" if ad_p <= 0.001 else ("> 0.250" if ad_p >= 0.25 else f"{ad_p:.3f}")
    except Exception:
        results["AD p-value"] = "Error"
        
    return results

def apply_custom_style(ax, title, xlabel, ylabel, xlim, bg, fg, grid, draw_legend=True):
    ax.set_facecolor(bg)
    ax.set_title(title, color=fg, fontsize=title_size)
    ax.set_xlabel(xlabel, color=fg, fontsize=label_size)
    ax.set_ylabel(ylabel, color=fg, fontsize=label_size)
    if xlim is not None: ax.set_xlim(xlim)
    
    # Apply tick size and axis box thickness
    ax.tick_params(colors=fg, labelsize=label_size, width=axes_width)
    for spine in ax.spines.values(): 
        spine.set_color(fg)
        spine.set_linewidth(axes_width)
        
    if grid: ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color=fg)
    
    # Apply legend formatting
    if draw_legend:
        # Check if there is data to label to avoid the empty square bug
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Save the title if Seaborn already created one
            existing_leg = ax.get_legend()
            leg_title = existing_leg.get_title().get_text() if existing_leg and existing_leg.get_title() else None
            
            # Rebuild the legend in the user's chosen location
            leg = ax.legend(loc=legend_position, fontsize=legend_size)
            
            if leg is not None:
                if leg_title: 
                    leg.set_title(leg_title)
                for text in leg.get_texts(): 
                    text.set_color(fg)
                    text.set_fontsize(legend_size)
                if leg.get_title(): 
                    leg.get_title().set_color(fg)
                    leg.get_title().set_fontsize(legend_size)
                leg.get_frame().set_facecolor(bg)
                leg.get_frame().set_edgecolor(fg)
                leg.get_frame().set_linewidth(axes_width)
    else:
        # If "Show Legend" is unchecked, actively delete any existing legends
        if ax.get_legend() is not None:
            ax.get_legend().remove()

def plot_custom_distribution(ax, data, feature_col, bins, x_vals, bin_width, color, style, line_width, label, display_style):
    x_data = data[feature_col].dropna()
    if len(x_data) == 0: return
    
    hist_lbl = label if display_style == "Histogram Only" else None
    curve_lbl = label if display_style in ["Smooth Curve Only", "Both (Bar + Curve)"] else None
    
    if display_style in ["Histogram Only", "Both (Bar + Curve)"]:
        ax.hist(x_data, bins=bins, histtype='step', color=color, linestyle=style, linewidth=line_width, label=hist_lbl)
        if display_style == "Both (Bar + Curve)":
            ax.hist(x_data, bins=bins, histtype='stepfilled', color=color, alpha=0.1)
            
    if display_style in ["Smooth Curve Only", "Both (Bar + Curve)"]:
        if len(x_data) > 1:
            y_counts = gaussian_kde(x_data)(x_vals) * len(x_data) * bin_width
            ax.plot(x_vals, y_counts, color=color, linestyle=style, linewidth=line_width, label=curve_lbl)

def plot_central_marker(ax, data, color, marker_type):
    val = np.nan
    if marker_type == "Median": val = data.median()
    elif marker_type == "Mode": val = get_mode_from_kde(data)
    elif marker_type == "Mean": val = data.mean()
    
    if pd.notna(val):
        ax.axvline(val, color=color, linestyle=':', linewidth=1.5, alpha=0.8)

def create_download_buttons(fig, filename_prefix):
    dl_col1, dl_col2 = st.columns(2)
    
    # Generate PNG
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", bbox_inches="tight", dpi=300, transparent=True)
    buf_png.seek(0)
    with dl_col1:
        st.download_button(
            label="📥 PNG",
            data=buf_png,
            file_name=f"{filename_prefix}.png",
            mime="image/png",
            use_container_width=True
        )
        
    # Generate SVG
    buf_svg = io.BytesIO()
    fig.savefig(buf_svg, format="svg", bbox_inches="tight", transparent=True)
    buf_svg.seek(0)
    with dl_col2:
        st.download_button(
            label="📥 SVG",
            data=buf_svg,
            file_name=f"{filename_prefix}.svg",
            mime="image/svg+xml",
            use_container_width=True
        )

# --- Main App Layout & Data Processing ---
if not uploaded_files:
    st.info("Please upload your ZetaSphere .zip files in the sidebar to begin analysis.")
else:
    processed_data = {'Size': {}, 'Zeta_Potential': {}, 'Concentration': {}, 'Colocalization': {}}
    for file in uploaded_files:
        measurement, channel = parse_file_info(file)
        df = extract_dataframe(file)
        if df is not None:
            if channel not in processed_data.get(measurement, {}): processed_data[measurement][channel] = []
            default_color = '#000000'
            for raw_code, (name, hex_code) in DEFAULT_CHANNELS.items():
                if name == channel: default_color = hex_code
            processed_data[measurement][channel].append({
                'filename': file.name, 'df': df, 'label': f"{channel} (Sample {len(processed_data[measurement][channel]) + 1})",
                'color': default_color, 'active': True, 'dilution': 1.0
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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Multi-Sample Size", "Zeta Potential", "Concentration", "Population Analysis (GMM)", "Colocalization", "Colocalization Quality"])

    # ==========================================
    # TAB 1: SIZE OVERVIEW & MULTI-SAMPLE
    # ==========================================
    with tab1:
        st.header("Size Distribution & Channel Selection")
        if not processed_data['Size']:
            st.warning("No 'Size' measurement files detected.")
        else:
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
                bins = np.arange(0, max_x_size + size_bin_width, size_bin_width)
                x_vals = np.linspace(0, max_x_size, 500)
                line_styles = ['-', '--', ':', '-.']
                
                entities = []
                for ch in active_channels:
                    active_items = [i for i in processed_data['Size'][ch] if i['active']]
                    if not active_items: continue
                    
                    if rep_mode == "Pool Data":
                        combined_df = pd.concat([i['df'] for i in active_items], ignore_index=True)
                        f_col = find_data_column(combined_df, ['particle size', 'size', 'diameter'])
                        if f_col:
                            entities.append({"label": f"{ch} (Pooled)", "data": combined_df[f_col].dropna(), "color": active_items[0]['color'], "style": '-'})
                    else:
                        for idx, item in enumerate(active_items):
                            f_col = find_data_column(item['df'], ['particle size', 'size', 'diameter'])
                            style = '-' if force_solid else line_styles[idx % len(line_styles)]
                            if f_col:
                                entities.append({"label": item['label'], "data": item['df'][f_col].dropna(), "color": item['color'], "style": style})

                # --- FIXED: Maximize color contrast for continuous palettes ---
                if entities and palette_choice != "Custom (Sample Colors)":
                    import matplotlib.colors as mcolors
                    import matplotlib.cm as cm
                    try:
                        cmap = plt.get_cmap(PALETTES[palette_choice])
                        # Sample from 10% to 95% to avoid the pitch-black and pure-white extremes
                        color_vals = np.linspace(0.10, 0.95, len(entities))
                        hex_colors = [mcolors.to_hex(cmap(c)) for c in color_vals]
                    except:
                        # Fallback for discrete lists of colors
                        hex_colors = sns.color_palette(PALETTES[palette_choice], len(entities)).as_hex()
                    
                    for i, ent in enumerate(entities): 
                        ent["color"] = hex_colors[i]

                if not entities:
                    st.info("No active data to plot.")
                else:
                    if multi_layout == "Overlay (Default)":
                        fig, ax = plt.subplots(figsize=(10, 5))
                        fig.patch.set_facecolor(bg_color)
                        for ent in entities:
                            tmp_df = pd.DataFrame({'val': ent['data']})
                            plot_custom_distribution(ax, tmp_df, 'val', bins, x_vals, size_bin_width, ent['color'], ent['style'], line_width, ent['label'], display_style)
                            if central_marker != "None": plot_central_marker(ax, ent['data'], ent['color'], central_marker)
                        apply_custom_style(ax, "Size Distribution", "Hydrodynamic Diameter (nm)", "Count", (0, max_x_size), bg_color, axes_color, show_grid, draw_legend=show_legend)
                        st.pyplot(fig)
                        create_download_buttons(fig, "Size_Distribution")

                    elif multi_layout == "Facet Grid":
                        cols = 2
                        rows = math.ceil(len(entities) / cols)
                        if rows == 0: rows = 1
                        fig, axes = plt.subplots(rows, cols, figsize=(10, max(4, rows * 3.5)), squeeze=False, sharey=facet_share_y)
                        fig.patch.set_facecolor(bg_color)
                        
                        for i, ent in enumerate(entities):
                            ax = axes[i // cols, i % cols]
                            tmp_df = pd.DataFrame({'val': ent['data']})
                            plot_custom_distribution(ax, tmp_df, 'val', bins, x_vals, size_bin_width, ent['color'], ent['style'], line_width, ent['label'], display_style)
                            if central_marker != "None": plot_central_marker(ax, ent['data'], ent['color'], central_marker)
                            apply_custom_style(ax, ent['label'], "Hydrodynamic Diameter (nm)", "Count", (0, max_x_size), bg_color, axes_color, show_grid, draw_legend=False)
                            
                        for j in range(len(entities), rows * cols): fig.delaxes(axes.flatten()[j])
                        plt.tight_layout()
                        st.pyplot(fig)
                        create_download_buttons(fig, "Size_Distribution")

                    elif multi_layout == "Ridgeline (Joyplot)":
                        fig, ax = plt.subplots(figsize=(10, max(5, len(entities) * 0.85)))
                        fig.patch.set_facecolor(bg_color)
                        ax.set_facecolor(bg_color)
                        
                        max_h = 0
                        kdes = []
                        for ent in entities:
                            if len(ent['data']) > 1:
                                y_c = gaussian_kde(ent['data'])(x_vals) * len(ent['data']) * size_bin_width
                                max_h = max(max_h, max(y_c))
                                kdes.append((y_c, ent))
                            else:
                                kdes.append((None, ent))
                                
                        offset_step = max_h * 0.4 if max_h > 0 else 1 
                        y_ticks, y_labels = [], []
                        
                        for i, (y_c, ent) in enumerate(reversed(kdes)):
                            current_offset = i * offset_step
                            if y_c is not None:
                                ax.fill_between(x_vals, current_offset, y_c + current_offset, color=ent['color'], alpha=0.6)
                                ax.plot(x_vals, y_c + current_offset, color=ent['color'], lw=line_width)
                                if central_marker != "None":
                                    val = np.nan
                                    if central_marker == "Median": val = ent['data'].median()
                                    elif central_marker == "Mode": val = get_mode_from_kde(ent['data'])
                                    elif central_marker == "Mean": val = ent['data'].mean()
                                    if pd.notna(val):
                                        ax.vlines(val, current_offset, current_offset + max(y_c), color=ent['color'], linestyle=':', lw=1.5, alpha=0.8)
                                        
                            ax.axhline(current_offset, color=axes_color, lw=0.5, alpha=0.4)
                            y_ticks.append(current_offset)
                            y_labels.append(ent['label'])
                            
                        ax.set_yticks(y_ticks)
                        ax.set_yticklabels(y_labels, color=axes_color)
                        apply_custom_style(ax, "Ridgeline Size Comparison", "Hydrodynamic Diameter (nm)", "", (0, max_x_size), bg_color, axes_color, show_grid, draw_legend=False)
                        st.pyplot(fig)
                        create_download_buttons(fig, "Size_Distribution")

                    elif multi_layout == "Violin Plot":
                        # Prepare data into a single DataFrame for Seaborn
                        violin_data = []
                        palette_dict = {}
                        for ent in entities:
                            tmp_df = pd.DataFrame({'Size': ent['data'], 'Sample': ent['label']})
                            violin_data.append(tmp_df)
                            palette_dict[ent['label']] = ent['color']
                            
                        df_violin = pd.concat(violin_data, ignore_index=True)
                        
                        # FIXED: Use a standard landscape width (10) so Streamlit doesn't stretch the height!
                        fig, ax = plt.subplots(figsize=(10, 6))
                        fig.patch.set_facecolor(bg_color)
                        
                        # Use quartiles for a clean, transparent inner look
                        if central_marker in ["Median", "Mean", "Mode"]: 
                            inner_style = "quart" 
                        else: 
                            inner_style = None
                        
                        import seaborn as sns
                        sns.violinplot(
                            data=df_violin, 
                            x='Sample', 
                            y='Size',
                            hue='Sample', # FIXED: Forces proper color mapping and builds legend handles
                            palette=palette_dict, 
                            inner=inner_style,
                            linewidth=line_width,
                            ax=ax
                        )
                        
                        # Style the axes for vertical orientation
                        ax.set_xlabel("")
                        ax.tick_params(colors=axes_color, labelsize=label_size)
                        
                        # Rotate sample names if they are long so they don't overlap
                        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                        
                        # FIXED: Pass 'show_legend' to the styling function
                        apply_custom_style(ax, "Violin Plot Size Comparison", "", "Hydrodynamic Diameter (nm)", None, bg_color, axes_color, show_grid, draw_legend=show_legend)
                        
                        # Limit the Y-axis to your sidebar slider max
                        ax.set_ylim(0, max_x_size)
                        
                        # Apply custom styling (Tell it NOT to draw the broken legend)
                        apply_custom_style(ax, "Violin Plot Size Comparison", "", "Hydrodynamic Diameter (nm)", None, bg_color, axes_color, show_grid, draw_legend=False)
                        
                        # --- FIXED: Manually construct a perfect color-coded legend ---
                        if show_legend:
                            import matplotlib.patches as mpatches
                            leg_size = legend_size if 'legend_size' in locals() else label_size
                            
                            # Create a colorful square (patch) for every single sample
                            legend_patches = [mpatches.Patch(facecolor=ent['color'], edgecolor=axes_color, label=ent['label']) for ent in entities]
                            
                            # Draw it!
                            leg = ax.legend(handles=legend_patches, loc=legend_position, facecolor=bg_color, edgecolor=axes_color, labelcolor=axes_color, fontsize=leg_size)
                            leg.get_frame().set_linewidth(axes_width)
                        
                        st.pyplot(fig)
                        create_download_buttons(fig, "Size_Distribution_Violin")

            st.markdown("---")
            st.subheader("📊 Statistical Comparison (Size)")
            if entities:
                summary_list = []
                for ent in entities:
                    d = ent["data"]
                    summary_list.append({"Sample": ent["label"], "Count": len(d), "Mean (nm)": round(d.mean(), 1), "Median (nm)": round(d.median(), 1), "Mode (nm)": round(get_mode_from_kde(d), 1), "SD (nm)": round(d.std(), 1), "MAD (nm)": round(get_mad(d), 1), "Span": round(get_span(d), 3), "FWHM (nm)": round(get_fwhm_from_kde(d), 1)})
                st.dataframe(pd.DataFrame(summary_list), use_container_width=True)
                
                if len(entities) > 1:
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
                        row_data.update(run_stats_comparisons(ent_a["data"], ent_b["data"]))
                        pairwise_list.append(row_data)
                    st.dataframe(pd.DataFrame(pairwise_list), use_container_width=True)

    # ==========================================
    # TAB 2: ZETA POTENTIAL
    # ==========================================
    with tab2:
        st.header("Zeta Potential Histogram")
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
                    
                st.markdown("---")
                show_marginal = st.checkbox("Show Size vs. Zeta Scatter Plot (Marginal Distributions)", value=False)
                if show_marginal:
                    bubble_mode = st.checkbox("Bubble Chart (3rd Variable)", value=False)
                    if not bubble_mode:
                        scatter_dot_size = st.slider("Scatter Dot Size", min_value=5, max_value=200, value=50, step=5)
                    else:
                        scatter_dot_size = 50 # Fallback
                
            with col4:
                bins = np.arange(min_x_zeta, max_x_zeta + zeta_bin_width, zeta_bin_width)
                x_vals = np.linspace(min_x_zeta, max_x_zeta, 500)
                line_styles = ['-', '--', ':', '-.']
                
                z_entities = []
                for ch in active_zeta_channels:
                    active_items = [i for i in processed_data['Zeta_Potential'][ch] if i['active']]
                    if not active_items: continue
                    
                    if rep_mode_zeta == "Pool Data":
                        combined_df = pd.concat([i['df'] for i in active_items], ignore_index=True)
                        f_col = find_data_column(combined_df, ['zeta potential'])
                        if f_col:
                            z_entities.append({"label": f"{ch} (Pooled)", "data": combined_df[f_col].dropna(), "df": combined_df, "color": active_items[0]['color'], "style": '-'})
                    else:
                        for idx, item in enumerate(active_items):
                            f_col = find_data_column(item['df'], ['zeta potential'])
                            style = '-' if force_solid else line_styles[idx % len(line_styles)]
                            if f_col:
                                z_entities.append({"label": item['label'], "data": item['df'][f_col].dropna(), "df": item['df'], "color": item['color'], "style": style})

                if z_entities and palette_choice != "Custom (Sample Colors)":
                    hex_colors = sns.color_palette(PALETTES[palette_choice], len(z_entities)).as_hex()
                    for i, ent in enumerate(z_entities): ent["color"] = hex_colors[i]

                if not z_entities:
                    st.info("No active data to plot.")
                else:
                    if multi_layout == "Overlay (Default)":
                        fig_zeta, ax_zeta = plt.subplots(figsize=(10, 5))
                        fig_zeta.patch.set_facecolor(bg_color)
                        for ent in z_entities:
                            tmp_df = pd.DataFrame({'val': ent['data']})
                            plot_custom_distribution(ax_zeta, tmp_df, 'val', bins, x_vals, zeta_bin_width, ent['color'], ent['style'], line_width, ent['label'], display_style)
                            if central_marker != "None": plot_central_marker(ax_zeta, ent['data'], ent['color'], central_marker)
                        apply_custom_style(ax_zeta, "Zeta Potential Distribution", "Zeta Potential (mV)", "Count", (min_x_zeta, max_x_zeta), bg_color, axes_color, show_grid, draw_legend=show_legend)
                        st.pyplot(fig_zeta)
                        create_download_buttons(fig_zeta, "Zeta_Potential_Histogram")

                    elif multi_layout == "Facet Grid":
                        cols = 2
                        rows = math.ceil(len(z_entities) / cols)
                        if rows == 0: rows = 1
                        fig_zeta, axes = plt.subplots(rows, cols, figsize=(10, max(4, rows * 3.5)), squeeze=False, sharey=facet_share_y)
                        fig_zeta.patch.set_facecolor(bg_color)
                        for i, ent in enumerate(z_entities):
                            ax = axes[i // cols, i % cols]
                            tmp_df = pd.DataFrame({'val': ent['data']})
                            plot_custom_distribution(ax, tmp_df, 'val', bins, x_vals, zeta_bin_width, ent['color'], ent['style'], line_width, ent['label'], display_style)
                            if central_marker != "None": plot_central_marker(ax, ent['data'], ent['color'], central_marker)
                            apply_custom_style(ax, ent['label'], "Zeta Potential (mV)", "Count", (min_x_zeta, max_x_zeta), bg_color, axes_color, show_grid, draw_legend=False)
                        for j in range(len(z_entities), rows * cols): fig_zeta.delaxes(axes.flatten()[j])
                        plt.tight_layout()
                        st.pyplot(fig_zeta)
                        create_download_buttons(fig_zeta, "Zeta_Potential_Histogram")
                        
                    elif multi_layout == "Ridgeline (Joyplot)":
                        fig_zeta, ax_zeta = plt.subplots(figsize=(10, max(5, len(z_entities) * 0.85)))
                        fig_zeta.patch.set_facecolor(bg_color)
                        ax_zeta.set_facecolor(bg_color)
                        
                        max_h = 0
                        kdes = []
                        for ent in z_entities:
                            if len(ent['data']) > 1:
                                y_c = gaussian_kde(ent['data'])(x_vals) * len(ent['data']) * zeta_bin_width
                                max_h = max(max_h, max(y_c))
                                kdes.append((y_c, ent))
                            else:
                                kdes.append((None, ent))
                                
                        offset_step = max_h * 0.4 if max_h > 0 else 1
                        y_ticks, y_labels = [], []
                        
                        for i, (y_c, ent) in enumerate(reversed(kdes)):
                            current_offset = i * offset_step
                            if y_c is not None:
                                ax_zeta.fill_between(x_vals, current_offset, y_c + current_offset, color=ent['color'], alpha=0.6)
                                ax_zeta.plot(x_vals, y_c + current_offset, color=ent['color'], lw=line_width)
                                if central_marker != "None":
                                    val = np.nan
                                    if central_marker == "Median": val = ent['data'].median()
                                    elif central_marker == "Mode": val = get_mode_from_kde(ent['data'])
                                    elif central_marker == "Mean": val = ent['data'].mean()
                                    if pd.notna(val):
                                        ax_zeta.vlines(val, current_offset, current_offset + max(y_c), color=ent['color'], linestyle=':', lw=1.5, alpha=0.8)
                            ax_zeta.axhline(current_offset, color=axes_color, lw=0.5, alpha=0.4)
                            y_ticks.append(current_offset)
                            y_labels.append(ent['label'])
                            
                        ax_zeta.set_yticks(y_ticks)
                        ax_zeta.set_yticklabels(y_labels, color=axes_color)
                        apply_custom_style(ax_zeta, "Ridgeline Zeta Comparison", "Zeta Potential (mV)", "", (min_x_zeta, max_x_zeta), bg_color, axes_color, show_grid, draw_legend=False)
                        st.pyplot(fig_zeta)
                        create_download_buttons(fig_zeta, "Zeta_Potential_Histogram")

                # --- Marginal Scatter Plot Logic ---
                if show_marginal and z_entities:
                    st.markdown("### Size vs. Zeta Potential Scatter Plot")
                    combined_marg_df = pd.DataFrame()
                    palette_dict = {}
                    
                    for ent in z_entities:
                        temp_df = ent['df'].copy()
                        temp_df['Sample'] = ent['label']
                        palette_dict[ent['label']] = ent['color']
                        combined_marg_df = pd.concat([combined_marg_df, temp_df], ignore_index=True)
                        
                    s_col = find_data_column(combined_marg_df, ['particle size', 'size', 'diameter'])
                    z_col = find_data_column(combined_marg_df, ['zeta potential'])
                    
                    if s_col and z_col:
                        if bubble_mode:
                            # Automatically find remaining numeric columns
                            numeric_cols = combined_marg_df.select_dtypes(include=[np.number]).columns.tolist()
                            valid_bubble_cols = [c for c in numeric_cols if c.lower() not in [s_col.lower(), z_col.lower(), 'channel']]
                            
                            if valid_bubble_cols:
                                bubble_col = st.selectbox("Select Variable for Bubble Size:", valid_bubble_cols)
                                jg = sns.jointplot(data=combined_marg_df, x=s_col, y=z_col, hue='Sample', palette=palette_dict, alpha=0.5, marginal_kws=dict(fill=True), joint_kws={'size': combined_marg_df[bubble_col], 'sizes': (20, 500)})
                            else:
                                st.warning("No suitable 3rd numeric variable found.")
                                jg = sns.jointplot(data=combined_marg_df, x=s_col, y=z_col, hue='Sample', palette=palette_dict, alpha=0.6, marginal_kws=dict(fill=True), joint_kws={'s': scatter_dot_size})
                        else:
                            jg = sns.jointplot(data=combined_marg_df, x=s_col, y=z_col, hue='Sample', palette=palette_dict, alpha=0.6, marginal_kws=dict(fill=True), joint_kws={'s': scatter_dot_size})
                            
                        jg.fig.patch.set_facecolor(bg_color)
                        jg.ax_joint.set_facecolor(bg_color)
                        jg.ax_marg_x.set_facecolor(bg_color)
                        jg.ax_marg_y.set_facecolor(bg_color)
                        
                        jg.ax_joint.set_xlabel("Hydrodynamic Diameter (nm)", color=axes_color, fontsize=label_size)
                        jg.ax_joint.set_ylabel("Zeta Potential (mV)", color=axes_color, fontsize=label_size)
                        jg.ax_joint.tick_params(colors=axes_color, labelsize=label_size, width=axes_width)
                        for spine in jg.ax_joint.spines.values(): 
                            spine.set_color(axes_color)
                            spine.set_linewidth(axes_width)
                            
                        # Apply thickness and colors to the marginal histogram axes
                        for ax_marg in [jg.ax_marg_x, jg.ax_marg_y]:
                            ax_marg.tick_params(colors=axes_color, width=axes_width)
                            for spine in ax_marg.spines.values():
                                spine.set_color(axes_color)
                                spine.set_linewidth(axes_width)
                        if show_grid: jg.ax_joint.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color=axes_color)
                        
                        if show_legend and jg.ax_joint.get_legend():
                            leg = jg.ax_joint.get_legend()
                            for text in leg.get_texts(): 
                                text.set_color(axes_color)
                                text.set_fontsize(legend_size)
                            if leg.get_title():
                                leg.get_title().set_color(axes_color)
                                leg.get_title().set_fontsize(legend_size)
                            leg.get_frame().set_facecolor(bg_color)
                            leg.get_frame().set_edgecolor(axes_color)
                            
                        elif not show_legend and jg.ax_joint.get_legend():
                            jg.ax_joint.get_legend().remove()

                        st.pyplot(jg.fig)
                        create_download_buttons(jg.fig, "Size_vs_Zeta_Scatter")
                    else:
                        st.error("Could not find both Size and Zeta Potential columns in the selected files to create the scatter plot.")

            st.markdown("---")
            st.subheader("📊 Statistical Comparison (Zeta Potential)")
            if z_entities:
                summary_list = [{"Sample": ent["label"], "Count": len(ent["data"]), "Mean (mV)": round(ent["data"].mean(), 1), "SD (mV)": round(ent["data"].std(), 1)} for ent in z_entities]
                st.dataframe(pd.DataFrame(summary_list), use_container_width=True)
                
                if len(z_entities) > 1:
                    st.info("""
                    **📚 Understanding the Statistical Tests:**
                    * **Mann-Whitney U:** Gives a p-value for the shift in the median.
                    * **Anderson-Darling:** Gives a p-value testing if the overall shape and tails of the distributions are different.
                    * **Earth Mover's Distance (EMD):** This does not yield a p-value. Instead, it gives a physical "Distance Score" (in nm or mV). A score of 0.0 means the histograms are identical. The higher the number, the more physical "work" is required to make one curve look like the other.
                    """)
                    st.markdown("*All-vs-All Pairwise Comparisons*")
                    pairwise_list = []
                    for ent_a, ent_b in itertools.combinations(z_entities, 2):
                        row_data = {"Sample A": ent_a["label"], "Sample B": ent_b["label"]}
                        row_data.update(run_stats_comparisons(ent_a["data"], ent_b["data"]))
                        pairwise_list.append(row_data)
                    st.dataframe(pd.DataFrame(pairwise_list), use_container_width=True)

    # ==========================================
    # TAB 3: CONCENTRATION
    # ==========================================
    with tab3:
        st.header("Concentration Analysis")
        st.info("💡 **Note:** The dilution factor (df) can typically be found at the end of the sample's original folder name (e.g., _df250000).")
        
        if not processed_data.get('Concentration'):
            st.warning("No 'Concentration' measurement files detected.")
        else:
            available_conc_channels = list(processed_data['Concentration'].keys())
            active_conc_channels = st.multiselect("Active Concentration Channels:", available_conc_channels, default=available_conc_channels, key="c_toggle")
            
            col_c1, col_c2 = st.columns([1, 3])
            with col_c1:
                fig_width = st.slider("Figure Width", min_value=3.0, max_value=15.0, value=6.0, step=0.5, help="Reduce this value to make the graph squarer when you have very few samples.")
                bar_width = st.slider("Bar/Box Width", min_value=0.1, max_value=1.0, value=0.8, step=0.1, help="Adjust the thickness of the individual bars or boxes.")
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
                        c_val = len(item['df']) * item['dilution']
                        conc_data.append({'Channel': ch, 'Sample Label': item['label'], 'Concentration (particles/mL)': c_val, 'Color': item['color']})
                        palette_dict[ch if grouping == "Sample (Compare Channels)" else item['label']] = item['color']
                            
                if conc_data:
                    df_conc = pd.DataFrame(conc_data)
                    fig_conc, ax_conc = plt.subplots(figsize=(fig_width, 5))
                    fig_conc.patch.set_facecolor(bg_color)
                    ax_conc.set_facecolor(bg_color)
                    
                    x_col = 'Channel' if grouping == "Channel (Compare Samples)" else 'Sample Label'
                    hue_col = 'Sample Label' if grouping == "Channel (Compare Samples)" else 'Channel'
                    
                    active_palette = palette_dict if palette_choice == "Custom (Sample Colors)" else PALETTES[palette_choice]
                    
                    if graph_type == "Bar Chart": sns.barplot(data=df_conc, x=x_col, y='Concentration (particles/mL)', hue=hue_col, palette=active_palette, errorbar='sd', capsize=0.1, ax=ax_conc, edgecolor=axes_color, linewidth=line_width, width=bar_width)
                    elif graph_type == "Dot Plot (Strip)": sns.stripplot(data=df_conc, x=x_col, y='Concentration (particles/mL)', hue=hue_col, palette=active_palette, dodge=True, size=8, ax=ax_conc, edgecolor=axes_color, linewidth=line_width/2)
                    elif graph_type == "Box Plot": sns.boxplot(data=df_conc, x=x_col, y='Concentration (particles/mL)', hue=hue_col, palette=active_palette, ax=ax_conc, fliersize=5, linewidth=line_width, width=bar_width)
                    
                    apply_custom_style(ax_conc, "Total Particle Concentration", x_col, "Concentration (particles/mL)", None, bg_color, axes_color, show_grid, draw_legend=show_legend)
                    
                    # Ensure the legend title (Channel vs Sample) matches the styling
                    if show_legend and ax_conc.get_legend():
                        leg = ax_conc.get_legend()
                        leg.set_title(hue_col)
                        leg.get_title().set_color(axes_color)
                        leg.get_title().set_fontsize(legend_size)
                    st.pyplot(fig_conc)
                    create_download_buttons(fig_conc, "Concentration_Plot")

            st.markdown("---")
            st.subheader("📊 Concentration Data")
            if conc_data:
                df_summary = df_conc.groupby(['Channel', 'Sample Label']).agg(Replicates=('Concentration (particles/mL)', 'count'), Mean_Concentration=('Concentration (particles/mL)', 'mean'), SD_Concentration=('Concentration (particles/mL)', 'std')).reset_index()
                df_summary['Mean_Concentration'] = df_summary['Mean_Concentration'].apply(lambda x: f"{x:.2e}")
                df_summary['SD_Concentration'] = df_summary['SD_Concentration'].fillna(0).apply(lambda x: f"{x:.2e}")
                st.dataframe(df_summary, use_container_width=True)

    # ==========================================
    # TAB 4: POPULATION ANALYSIS (GMM)
    # ==========================================
    with tab4:
        st.header("Advanced Population Analysis")
        st.markdown("This module calculates GMM sub-populations, stats, and positivity rates for activated channels.")
        
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
                        
                        summary = df_clean.groupby('Population')[feature_col].agg(
                            Mean='mean', Median=np.median, Mode=get_mode_from_kde, 
                            SD='std', MAD=get_mad, Count='count', Span=get_span, FWHM=get_fwhm_from_kde
                        )
                        summary['Percentage (%)'] = (summary['Count'] / total_particles) * 100
                        
                        summary['Mean (nm)'] = summary['Mean'].round(1)
                        summary['Median (nm)'] = summary['Median'].round(1)
                        summary['Mode (nm)'] = summary['Mode'].round(1)
                        summary['SD (nm)'] = summary['SD'].round(1)
                        summary['MAD (nm)'] = summary['MAD'].round(1)
                        summary['Span'] = summary['Span'].round(3)
                        summary['FWHM (nm)'] = summary['FWHM'].round(1)
                        summary = summary.drop(columns=['Mean', 'Median', 'Mode', 'SD', 'MAD', 'FWHM']).reset_index()
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
                        scatter_data = gmm_results['Scatter']['df'][gmm_results['Scatter']['feature_col']].dropna() if 'Scatter' in gmm_results else None
                        
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
                                stat_results = run_stats_comparisons(scatter_data, ch_data)
                            
                            global_row = {'Channel': ch, 'Total Particle Count': ch_count, 'Global Median Size (nm)': round(ch_median, 2), 'Positivity Rate': pos_rate_str}
                            global_row.update(stat_results)
                            global_stats_list.append(global_row)
                        
                        st.markdown("### 📊 Global Overview")
                        if len(gmm_results) > 1: st.caption("* Positivity rate is calculated by particle count ratios, not true co-localization.")
                        st.dataframe(pd.DataFrame(global_stats_list), use_container_width=True)
                        
                        st.markdown("### 🧬 Sub-Population Details")
                        for ch in gmm_results:
                            st.markdown(f"**{ch} Channel Breakdown:**")
                            st.dataframe(gmm_results[ch]['summary'], use_container_width=True)
                        
                        min_val_ext, max_val_ext = 0, gmm_x_max
                        x_vals_ext = np.linspace(min_val_ext, max_val_ext, 500)
                        master_curves = pd.DataFrame({'Hydrodynamic Diameter (nm)': x_vals_ext})
                        bins_ext = np.arange(min_val_ext, max_val_ext + gmm_bin_width, gmm_bin_width)
                        master_hists = pd.DataFrame({'Bin_Start (nm)': bins_ext[:-1], 'Bin_End (nm)': bins_ext[1:]})
                        
                        for ch, res in gmm_results.items():
                            df_ch, f_col, best_n = res['df'], res['feature_col'], res['best_n']
                            if len(df_ch) > 1: master_curves[f'{ch}_Total_Density'] = gaussian_kde(df_ch[f_col])(x_vals_ext) * len(df_ch)
                            master_hists[f'{ch}_Total_Count'], _ = np.histogram(df_ch[f_col], bins=bins_ext)
                            for pop in range(1, best_n + 1):
                                pop_data = df_ch[df_ch['Population'] == pop][f_col]
                                if len(pop_data) > 1: master_curves[f'{ch}_Pop_{pop}_Density'] = gaussian_kde(pop_data)(x_vals_ext) * len(pop_data)
                                counts_pop, _ = np.histogram(pop_data, bins=bins_ext)
                                master_hists[f'{ch}_Pop_{pop}_Count'] = counts_pop

                        first_ch = list(gmm_results.keys())[0]
                        dynamic_excel_name = f"{gmm_results[first_ch]['filenames'][0].replace('.zip', '')}_Population_Analysis.xlsx"
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
                        
                        if num_channels == 1:
                            # --- 1-CHANNEL LAYOUT (Strictly 1 Row) ---
                            fig_gmm = plt.figure(figsize=(14, 5))
                            fig_gmm.patch.set_facecolor(bg_color)
                            
                            ax1 = plt.subplot(1, 2, 1)
                            ax2 = plt.subplot(1, 2, 2)
                            
                            ch, res = list(gmm_results.items())[0]
                            ax1.plot(range(1, max_pops + 1), res['bic'], marker='o', linestyle='-', color=res['color'])
                            ax1.set_xticks(range(1, max_pops + 1))  
                            apply_custom_style(ax1, 'Model Scoring (Lowest BIC Wins)', 'Populations Tested', 'BIC Score', None, bg_color, axes_color, show_grid, draw_legend=False)
                            
                            # --- FIXED: Stop color repeating on discrete palettes ---
                            pops = np.sort(res['df']['Population'].unique())
                            palette_name = PALETTES[palette_choice] if palette_choice != "Custom (Sample Colors)" else "viridis"
                            
                            # Ask Seaborn for the EXACT number of colors needed. No complex math!
                            exact_colors = sns.color_palette(palette_name, n_colors=max(1, len(pops)))
                            pop_color_dict = {pop: exact_colors[idx] for idx, pop in enumerate(pops)}
                            
                            sns.histplot(data=res['df'], x=res['feature_col'], hue='Population', palette=pop_color_dict, element='step' if display_style != "Smooth Curve Only" else None, binwidth=gmm_bin_width, binrange=(0, gmm_x_max), kde=True, fill=display_style != "Smooth Curve Only", alpha=0.2 if display_style != "Smooth Curve Only" else 0, line_kws={'linewidth': line_width}, linewidth=line_width, ax=ax2)
                            
                            # Add Vertical Markers to Single Channel
                            if central_marker != "None":
                                for pop in pops:
                                    p_data = res['df'][res['df']['Population'] == pop][res['feature_col']].dropna()
                                    if len(p_data) > 1:
                                        val = None
                                        if central_marker == "Mean": val = p_data.mean()
                                        elif central_marker == "Median": val = p_data.median()
                                        elif central_marker == "Mode": val = get_mode_from_kde(p_data)
                                        
                                        if val is not None and not pd.isna(val):
                                            ax2.axvline(val, color=pop_color_dict[pop], linestyle='--', linewidth=line_width, zorder=5)
                            
                            apply_custom_style(ax2, f"{ch} Particles Grouped into {res['best_n']} Populations", "Hydrodynamic Diameter (nm)", "Count", (0, gmm_x_max), bg_color, axes_color, show_grid, draw_legend=show_legend)
                            
                        else:
                            # --- MULTI-CHANNEL LAYOUT (Strict Grid Layout) ---
                            total_rows = num_channels + 1
                            fig_gmm = plt.figure(figsize=(14, 5 * total_rows))
                            fig_gmm.patch.set_facecolor(bg_color)
                            
                            gs = fig_gmm.add_gridspec(total_rows, 2)
                            ax1 = fig_gmm.add_subplot(gs[0, 0])
                            ax2 = fig_gmm.add_subplot(gs[0, 1])
                            
                            for ch, res in gmm_results.items():
                                c = res['color']
                                ax1.plot(range(1, max_pops + 1), res['bic'], marker='o', linestyle='-', color=c, label=ch)
                                plot_custom_distribution(ax2, res['df'], res['feature_col'], bins_ext, x_vals_ext, gmm_bin_width, c, '-', line_width, ch, display_style)
                            
                            ax1.set_xticks(range(1, max_pops + 1)) 
                            apply_custom_style(ax1, 'Model Scoring (Lowest BIC Wins)', 'Populations Tested', 'BIC Score', None, bg_color, axes_color, show_grid, draw_legend=show_legend)
                            apply_custom_style(ax2, "Multi-Channel Size Distribution Overlay", "Hydrodynamic Diameter (nm)", "Count", (0, gmm_x_max), bg_color, axes_color, show_grid, draw_legend=show_legend)
                            
                            for i, (ch, res) in enumerate(gmm_results.items()):
                                ax_sub = fig_gmm.add_subplot(gs[i + 1, :])
                                
                                # --- FIXED: Stop color repeating on discrete palettes ---
                            pops = np.sort(res['df']['Population'].unique())
                            palette_name = PALETTES[palette_choice] if palette_choice != "Custom (Sample Colors)" else "viridis"
                            
                            # Ask Seaborn for the EXACT number of colors needed. No complex math!
                            exact_colors = sns.color_palette(palette_name, n_colors=max(1, len(pops)))
                            pop_color_dict = {pop: exact_colors[idx] for idx, pop in enumerate(pops)}
                                
                                # 1. The Updated Plot Generator
                                sns.histplot(
                                    data=res['df'], 
                                    x=res['feature_col'], 
                                    hue='Population', 
                                    palette=pop_color_dict, 
                                    element='step', 
                                    binwidth=gmm_bin_width, 
                                    binrange=(0, gmm_x_max), 
                                    kde=(display_style != "Histogram Only"), # <-- FIXED!
                                    fill=(display_style != "Smooth Curve Only"), 
                                    alpha=(0.2 if display_style != "Smooth Curve Only" else 0.0), 
                                    line_kws={'linewidth': line_width}, 
                                    linewidth=(line_width if display_style != "Smooth Curve Only" else 0), 
                                    ax=ax_sub
                                )
                    
                                # 2. Rebuild the legend for "Smooth Curve Only" so it isn't invisible
                                if display_style == "Smooth Curve Only":
                                    import matplotlib.lines as mlines
                                    legend = ax_sub.get_legend()
                                    if legend is not None:
                                        # Force all dictionary keys to strings to guarantee a match
                                        str_color_dict = {str(k).strip(): v for k, v in pop_color_dict.items()}

                                        handles, labels = [], []
                                        for text_obj in legend.get_texts():
                                            pop_name = text_obj.get_text().strip()
                                            labels.append(pop_name)
                                            # Grab the correct color using the string-matched dictionary
                                            color = str_color_dict.get(pop_name, axes_color)
                                            handles.append(mlines.Line2D([], [], color=color, linewidth=line_width))
                            
                                        # Overwrite the invisible legend with our new solid colored lines
                                        ax_sub.legend(handles=handles, labels=labels, title=legend.get_title().get_text())
                                
                                # Add Vertical Markers to Multi-Channel
                                if central_marker != "None":
                                    for pop in pops:
                                        p_data = res['df'][res['df']['Population'] == pop][res['feature_col']].dropna()
                                        if len(p_data) > 1:
                                            val = None
                                            if central_marker == "Mean": val = p_data.mean()
                                            elif central_marker == "Median": val = p_data.median()
                                            elif central_marker == "Mode": val = get_mode_from_kde(p_data)
                                            
                                            if val is not None and not pd.isna(val):
                                                ax_sub.axvline(val, color=pop_color_dict[pop], linestyle='--', linewidth=line_width, zorder=5)
                                                
                                apply_custom_style(ax_sub, f"Sub-population Breakdown: {ch} ({res['best_n']} Populations Found)", "Hydrodynamic Diameter (nm)", "Count", (0, gmm_x_max), bg_color, axes_color, show_grid, draw_legend=show_legend)

                        plt.tight_layout()
                        st.pyplot(fig_gmm)
                        create_download_buttons(fig_gmm, "GMM_Population_Analysis")
                        
                        st.download_button(
                            label="📥 Download Excel Statistics Data",
                            data=excel_buffer.getvalue(),
                            file_name=dynamic_excel_name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

    # ==========================================
    # TAB 5: COLOCALIZATION
    # ==========================================
    with tab5:
        st.header("Colocalization Analysis")
        use_palette_t5 = st.checkbox("🎨 Override default colors with sidebar palette", key="tab5_color_override")

        if not processed_data.get('Colocalization'):
            st.warning("No 'Colocalization' measurement files detected.")
        else:
            st.info("💡 **Note:** Pie charts display the percentage of unique particles colocalized versus particles detected strictly in a single channel.")
            
            active_coloc_items = []
            for ch, items in processed_data['Colocalization'].items():
                for item in items:
                    if item['active']:
                        active_coloc_items.append((ch, item))
            
            if not active_coloc_items:
                st.info("No active Colocalization data to plot.")
            else:
                # --- NEW: Tab 5 Override Checkbox ---
                
                coloc_summary = []
                plot_data_list = []  # Stores data to generate the master grid later
                cols = st.columns(3)
                
                for idx, (ch_group, item) in enumerate(active_coloc_items):
                    df = item['df']
                    ch_col = find_data_column(df, ['channel'])
                    coloc_col = find_data_column(df, ['colocalised', 'colocalized'])
                    
                    if ch_col and coloc_col:
                        name_lower = item['filename'].lower()
                        found = []
                        for code, (c_name, color) in DEFAULT_CHANNELS.items():
                            id_pos = name_lower.find(code.lower())
                            if id_pos != -1: found.append((id_pos, c_name, color))
                        found.sort()
                        
                        ch1_name = found[0][1] if len(found) >= 1 else "Channel 1"
                        ch2_name = found[1][1] if len(found) >= 2 else "Channel 2"
                        
                        # --- NEW: Conditional Color Logic ---
                        if use_palette_t5:
                            # Safely extract exactly 3 hex colors using the DICTIONARY VALUE
                            custom_colors = sns.color_palette(PALETTES[palette_choice], 3).as_hex()
                            ch1_color = custom_colors[0]
                            ch2_color = custom_colors[1]
                            coloc_color = custom_colors[2]
                        else:
                            ch1_color = found[0][2] if len(found) >= 1 else "#4169E1"
                            ch2_color = found[1][2] if len(found) >= 2 else "#228B22"
                            coloc_color = "#FFD700"
                        
                        is_coloc = df[coloc_col].astype(str).str.strip().str.upper().isin(['TRUE', '1', '1.0'])
                        
                        unique_chs = df[ch_col].dropna().unique()
                        ch1_val = unique_chs[0] if len(unique_chs) > 0 else 1
                        ch2_val = unique_chs[1] if len(unique_chs) > 1 else 2
                        
                        ch1_mask = df[ch_col] == ch1_val
                        ch2_mask = df[ch_col] == ch2_val
                        
                        coloc_events = max(df[is_coloc & ch1_mask].shape[0], df[is_coloc & ch2_mask].shape[0])
                        ch1_only = df[~is_coloc & ch1_mask].shape[0]
                        ch2_only = df[~is_coloc & ch2_mask].shape[0]
                        
                        total_unique = coloc_events + ch1_only + ch2_only
                        if total_unique == 0: continue
                        
                        sizes = [ch1_only, ch2_only, coloc_events]
                        labels = [f"Only {ch1_name}", f"Only {ch2_name}", "Colocalized"]
                        colors = [ch1_color, ch2_color, coloc_color]
                        
                        plot_sizes = [s for s in sizes if s > 0]
                        plot_labels = [l for s, l in zip(sizes, labels) if s > 0]
                        plot_colors = [c for s, c in zip(sizes, colors) if s > 0]
                        
                        # Save for the Master Grid
                        plot_data_list.append({
                            'title': item['label'], 'sizes': plot_sizes, 
                            'labels': plot_labels, 'colors': plot_colors
                        })
                        
                        fig, ax = plt.subplots(figsize=(6, 5) if show_legend else (5, 5))
                        fig.patch.set_facecolor(bg_color)
                        
                        # Safely define legend size
                        leg_size = legend_size if 'legend_size' in locals() else label_size

                        # Swap external text for a legend if the user has requested it
                        pie_labels = None if show_legend else plot_labels
                        
                        wedges, texts, autotexts = ax.pie(
                            plot_sizes, labels=pie_labels, autopct='%1.1f%%', colors=plot_colors, 
                            startangle=140, textprops={'color': axes_color, 'fontsize': label_size}, 
                            # --- FIXED: Use line_width instead of axes_width for pie slices ---
                            wedgeprops={'edgecolor': axes_color, 'linewidth': line_width}
                        )
                        
                        if show_legend:
                            # --- FIXED: Inject leg_size into the legend and axes_width to its frame ---
                            legend = ax.legend(wedges, plot_labels, loc="center left", bbox_to_anchor=(1, 0.5), facecolor=bg_color, edgecolor=axes_color, labelcolor=axes_color, fontsize=leg_size)
                            legend.get_frame().set_linewidth(axes_width)
                            
                        ax.set_title(item['label'], color=axes_color, fontweight='bold', fontsize=title_size)
                        
                        # Render pie chart and Individual Download Buttons inside the column
                        with cols[idx % 3]:
                            st.pyplot(fig)
                            safe_filename = f"Colocalization_{item['label'].replace(' ', '_')}"
                            create_download_buttons(fig, safe_filename)
                            plt.close(fig)
                        
                        coloc_summary.append({
                            "Sample Label": item['label'],
                            f"Only {ch1_name} Count": ch1_only,
                            f"Only {ch2_name} Count": ch2_only,
                            "Colocalized Count": coloc_events,
                            "Total Unique Particles": total_unique,
                            "Colocalized (%)": round((coloc_events / total_unique) * 100, 1) if total_unique > 0 else 0
                        })
                    else:
                        st.warning(f"File {item['filename']} is missing 'Channel' or 'Colocalised' columns.")
                        
                # --- NEW: Master Grid Generator ---
                if plot_data_list:
                    st.markdown("---")
                    st.subheader("🖼️ Download Master Grid")
                    st.markdown("Export all active pie charts combined into a single, high-resolution panel:")
                    
                    n_plots = len(plot_data_list)
                    cols_grid = min(3, n_plots)
                    rows_grid = (n_plots + cols_grid - 1) // cols_grid
                    
                    # Expand width slightly to accommodate legends without squeezing the circles
                    master_width = (6 if show_legend else 5) * cols_grid
                    fig_master = plt.figure(figsize=(master_width, 5 * rows_grid))
                    fig_master.patch.set_facecolor(bg_color)
                    
                    # Safely define legend size for master grid too
                    leg_size = legend_size if 'legend_size' in locals() else label_size
                    
                    for i, p_data in enumerate(plot_data_list):
                        ax_m = fig_master.add_subplot(rows_grid, cols_grid, i + 1)
                        pie_labels_m = None if show_legend else p_data['labels']
                        
                        wedges_m, texts_m, autotexts_m = ax_m.pie(
                            p_data['sizes'], labels=pie_labels_m, autopct='%1.1f%%', 
                            colors=p_data['colors'], startangle=140, 
                            textprops={'color': axes_color, 'fontsize': label_size},
                            # --- FIXED: Use line_width here too ---
                            wedgeprops={'edgecolor': axes_color, 'linewidth': line_width}
                        )
                        ax_m.set_title(p_data['title'], color=axes_color, fontweight='bold', fontsize=title_size)
                        
                        if show_legend:
                            # --- FIXED: Inject leg_size into the master grid legend ---
                            legend_m = ax_m.legend(wedges_m, p_data['labels'], loc="center left", bbox_to_anchor=(1, 0.5), facecolor=bg_color, edgecolor=axes_color, labelcolor=axes_color, fontsize=leg_size)
                            legend_m.get_frame().set_linewidth(axes_width)
                    
                    # Ensure legends don't get cut off from the edges of the image
                    plt.tight_layout()
                    create_download_buttons(fig_master, "Colocalization_Master_Grid")
                    plt.close(fig_master)
                        
                st.markdown("---")
                st.subheader("📊 Colocalization Data Summary")
                if coloc_summary:
                    st.dataframe(pd.DataFrame(coloc_summary), use_container_width=True)

    # ==========================================
    # TAB 6: COLOCALIZATION QUALITY
    # ==========================================
    with tab6:
        st.header("Colocalization Quality & Morphology")
        use_palette_t6 = st.checkbox("🎨 Override default colors with sidebar palette", key="tab6_color_override")
        
        if not processed_data.get('Colocalization'):
            st.warning("No 'Colocalization' measurement files detected.")
        else:
            active_coloc_items = [item for items in processed_data['Colocalization'].values() for item in items if item['active']]
            
            if not active_coloc_items:
                st.info("No active Colocalization data to plot.")
            else:
                for idx, item in enumerate(active_coloc_items):
                    df = item['df'].copy()
                    
                    # Find our necessary columns
                    ch_col = find_data_column(df, ['channel'])
                    coloc_col = find_data_column(df, ['colocalised', 'colocalized'])
                    pos_col = find_data_column(df, ['position'])
                    x_col = find_data_column(df, ['xc'])
                    y_col = find_data_column(df, ['yc'])
                    int_col = find_data_column(df, ['mean intensity'])
                    area_col = find_data_column(df, ['mean area'])
                    ar_col = find_data_column(df, ['aspect ratio'])
                    size_col = find_data_column(df, ['particle size'])
                    
                    if not all([ch_col, coloc_col, pos_col, x_col, y_col, int_col, area_col, ar_col, size_col]):
                        st.warning(f"File {item['filename']} is missing required morphology/coordinate columns.")
                        continue
                        
                    st.subheader(f"Data for: {item['label']}")
                    
                    # 1. Filter for ONLY colocalized particles
                    coloc_df = df[df[coloc_col].astype(str).str.strip().str.upper().isin(['TRUE', '1', '1.0'])]
                    
                    unique_chs = df[ch_col].dropna().unique()
                    if len(unique_chs) < 2: continue
                    ch1_val, ch2_val = unique_chs[0], unique_chs[1]
                    
                    matched_pairs = []
                    
                    # 2. Iterate by position to find matches
                    for pos in coloc_df[pos_col].unique():
                        pos_data = coloc_df[coloc_df[pos_col] == pos]
                        c1_data = pos_data[pos_data[ch_col] == ch1_val]
                        c2_data = pos_data[pos_data[ch_col] == ch2_val]
                        
                        if c1_data.empty or c2_data.empty: continue
                        
                        # 3. Euclidean Distance Matrix using NumPy
                        x1, y1 = c1_data[x_col].values, c1_data[y_col].values
                        x2, y2 = c2_data[x_col].values, c2_data[y_col].values
                        
                        dist_matrix = np.sqrt((x1[:, np.newaxis] - x2)**2 + (y1[:, np.newaxis] - y2)**2)
                        
                        # 4. Find pairs within the link radius
                        for i in range(dist_matrix.shape[0]):
                            min_idx = np.argmin(dist_matrix[i])
                            if dist_matrix[i, min_idx] <= link_radius:
                                matched_pairs.append({
                                    'C1_Intensity': c1_data.iloc[i][int_col],
                                    'C2_Intensity': c2_data.iloc[min_idx][int_col],
                                    'Colocalized_Area': (c1_data.iloc[i][area_col] + c2_data.iloc[min_idx][area_col]) / 2,
                                    'Colocalized_AR': (c1_data.iloc[i][ar_col] + c2_data.iloc[min_idx][ar_col]) / 2,
                                    'Colocalized_Size': (c1_data.iloc[i][size_col] + c2_data.iloc[min_idx][size_col]) / 2
                                })
                                
                    matched_df = pd.DataFrame(matched_pairs)
                    
                    if matched_df.empty:
                        st.warning("No colocalized pairs could be matched within the given Link Radius.")
                        continue
                        
                    # Create the 3-panel plotting grid
                    w = (fig_width * 3) if 'fig_width' in locals() else 18
                    h = fig_height if 'fig_height' in locals() else 5
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(w, h))
                    fig.patch.set_facecolor(bg_color)
                    
                    # Safely handle legend size if it exists in your sidebar
                    leg_size = legend_size if 'legend_size' in locals() else label_size
                    
                    # Plot 1: Intensity Stoichiometry (Scatter)
                    sns.regplot(data=matched_df, x='C1_Intensity', y='C2_Intensity', ax=ax1, scatter_kws={'alpha': 0.5}, color='#FFD700', line_kws={'linewidth': line_width})
                    ax1.set_title("Dye Stoichiometry", color=axes_color, fontweight='bold', fontsize=title_size)
                    ax1.set_xlabel("Channel 1 Mean Intensity", color=axes_color, fontsize=label_size)
                    ax1.set_ylabel("Channel 2 Mean Intensity", color=axes_color, fontsize=label_size)
                    
                    # --- NEW: DYNAMIC CHANNEL NAMES & COLORS ---
                    name_lower = item['filename'].lower()
                    found = []
                    for code, (c_name, color) in DEFAULT_CHANNELS.items():
                        id_pos = name_lower.find(code.lower())
                        if id_pos != -1: found.append((id_pos, c_name, color))
                    found.sort()
                    
                    ch1_name = found[0][1] if len(found) >= 1 else "Channel 1"
                    ch2_name = found[1][1] if len(found) >= 2 else "Channel 2"
                    
                    if use_palette_t6:
                        custom_colors = sns.color_palette(PALETTES[palette_choice], 3).as_hex()
                        ch1_color = custom_colors[0]
                        ch2_color = custom_colors[1]
                        coloc_color = custom_colors[2]
                    else:
                        ch1_color = found[0][2] if len(found) >= 1 else "#4169E1"
                        ch2_color = found[1][2] if len(found) >= 2 else "#228B22"
                        coloc_color = "#FFD700" 
                        
                    palette_colors = [ch1_color, ch2_color, coloc_color]

                    # Data Prep for Morphology & Size
                    single_c1 = df[(df[ch_col] == ch1_val) & (~df[coloc_col].astype(str).str.strip().str.upper().isin(['TRUE', '1', '1.0']))]
                    single_c2 = df[(df[ch_col] == ch2_val) & (~df[coloc_col].astype(str).str.strip().str.upper().isin(['TRUE', '1', '1.0']))]
                    
                    morph_df = pd.DataFrame({
                        'Group': [f'Only {ch1_name}']*len(single_c1) + [f'Only {ch2_name}']*len(single_c2) + ['Colocalized']*len(matched_df),
                        'Area': pd.concat([single_c1[area_col], single_c2[area_col], matched_df['Colocalized_Area']], ignore_index=True),
                        'Size': pd.concat([single_c1[size_col], single_c2[size_col], matched_df['Colocalized_Size']], ignore_index=True)
                    })
                    
                    # Plot 2: Aggregation Check (Boxplot for Area)
                    sns.boxplot(data=morph_df, x='Group', y='Area', ax=ax2, palette=palette_colors, linewidth=line_width)
                    ax2.set_title("Aggregation Check (Area)", color=axes_color, fontweight='bold', fontsize=title_size)
                    ax2.set_ylabel("Mean Area", color=axes_color, fontsize=label_size)
                    ax2.set_xlabel("", fontsize=label_size)
                    
                    # Plot 3: Hydrodynamic Size Shift (KDE)
                    sns.kdeplot(data=morph_df, x='Size', hue='Group', ax=ax3, fill=True, palette=palette_colors, alpha=0.3, linewidth=line_width, legend=show_legend)
                    ax3.set_title("Hydrodynamic Size Shift", color=axes_color, fontweight='bold', fontsize=title_size)
                    ax3.set_xlabel("Particle Size (nm)", color=axes_color, fontsize=label_size)
                    ax3.set_ylabel("Density", color=axes_color, fontsize=label_size)
                    
                    # Polish axes and typography
                    for ax in [ax1, ax2, ax3]:
                        ax.tick_params(colors=axes_color, labelsize=label_size, width=axes_width)
                        for spine in ax.spines.values(): 
                            spine.set_color(axes_color)
                            spine.set_linewidth(axes_width)
                        ax.set_facecolor(bg_color)
                        
                    # Style the legend for the KDE plot (tied to the sidebar toggle)
                    if show_legend:
                        legend = ax3.get_legend()
                        if legend is not None:
                            plt.setp(legend.get_texts(), color=axes_color, fontsize=leg_size)
                            plt.setp(legend.get_title(), color=axes_color, fontsize=leg_size, fontweight='bold')
                            legend.get_frame().set_facecolor(bg_color)
                            legend.get_frame().set_edgecolor(axes_color)
                            legend.get_frame().set_linewidth(axes_width)
                            
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # --- NEW: Image Download Buttons ---
                    safe_filename = f"Coloc_Quality_{item['label'].replace(' ', '_')}"
                    create_download_buttons(fig, safe_filename)
                    plt.close(fig)

                    # --- Stoichiometry Statistics & Scoring ---
                    r_val = matched_df['C1_Intensity'].corr(matched_df['C2_Intensity'])
                    r_sq = r_val ** 2
                    
                    if r_val >= 0.7:
                        score, exp = "🟢 Strong (Good)", "A high score indicates proportional binding. For TetraSpeck beads, this proves the fluorophores are evenly distributed on the particles. In biological samples, this suggests target receptors are expressed at constant ratios."
                    elif r_val >= 0.4:
                        score, exp = "🟡 Moderate (Medium)", "A medium score suggests some proportional binding, but with significant variation. Beads might be photobleaching unevenly, or biological targets have variable expression."
                    else:
                        score, exp = "🔴 Weak (Bad)", "A low score means the intensities are independent. Binding is random. For TetraSpeck beads, this indicates severe degradation, photobleaching, or measurement noise."

                    st.markdown("### 📈 Stoichiometry Statistical Summary")
                    stat_cols = st.columns(3)
                    stat_cols[0].metric("Pearson Correlation (r)", round(r_val, 3))
                    stat_cols[1].metric("R-squared (R²)", round(r_sq, 3))
                    stat_cols[2].metric("Correlation Quality", score)
                    st.info(f"**Interpretation:** {exp}")

                    # --- Morphology & Size Statistics ---
                    st.markdown("### 🔬 Aggregation & Size Summary")
                    
                    med_area_c1 = morph_df[morph_df['Group'] == f'Only {ch1_name}']['Area'].median()
                    med_area_c2 = morph_df[morph_df['Group'] == f'Only {ch2_name}']['Area'].median()
                    med_area_coloc = morph_df[morph_df['Group'] == 'Colocalized']['Area'].median()
                    
                    med_area_c1 = med_area_c1 if pd.notna(med_area_c1) else 0
                    med_area_c2 = med_area_c2 if pd.notna(med_area_c2) else 0
                    
                    avg_single_area = (med_area_c1 + med_area_c2) / 2
                    area_ratio = med_area_coloc / avg_single_area if avg_single_area > 0 else 1
                    
                    if area_ratio >= 1.5:
                        area_score, area_exp = "🔴 High Risk", "Colocalized particles have a significantly larger cross-sectional area (≥50% bigger) than single-positive particles. This strongly suggests physical clumping (doublets/aggregates) rather than true single-particle colocalization."
                    elif area_ratio >= 1.2:
                        area_score, area_exp = "🟡 Moderate Risk", "Colocalized particles are slightly larger on average, indicating a possible mix of true colocalized events and some small aggregates."
                    else:
                        area_score, area_exp = "🟢 Low Risk", "The optical areas are highly comparable. Colocalized particles maintain a single-particle optical profile, confirming high-quality colocalization without clumping."

                    st.markdown("**1. Aggregation Check (Area)**")
                    area_cols = st.columns(4)
                    area_cols[0].metric(f"{ch1_name} Median Area", round(med_area_c1, 1))
                    area_cols[1].metric(f"{ch2_name} Median Area", round(med_area_c2, 1))
                    area_cols[2].metric("Colocalized Area", round(med_area_coloc, 1))
                    area_cols[3].metric("Aggregation Status", area_score)
                    st.info(f"**Interpretation:** {area_exp}")

                    med_size_c1 = morph_df[morph_df['Group'] == f'Only {ch1_name}']['Size'].median()
                    med_size_c2 = morph_df[morph_df['Group'] == f'Only {ch2_name}']['Size'].median()
                    med_size_coloc = morph_df[morph_df['Group'] == 'Colocalized']['Size'].median()
                    
                    med_size_c1 = med_size_c1 if pd.notna(med_size_c1) else 0
                    med_size_c2 = med_size_c2 if pd.notna(med_size_c2) else 0
                    
                    avg_single_size = (med_size_c1 + med_size_c2) / 2
                    size_ratio = med_size_coloc / avg_single_size if avg_single_size > 0 else 1
                    
                    if size_ratio >= 1.2:
                        size_score, size_exp = "🔴 Shift Detected", "Colocalized particles have a notably larger hydrodynamic diameter. The dual-labeling may be inducing aggregation, or the dyes are selectively binding to larger particles in the overall population."
                    else:
                        size_score, size_exp = "🟢 Consistent Size", "Dual-labeled particles share a nearly identical hydrodynamic size with single-labeled particles, confirming that dual-labeling does not severely alter their physical profile."

                    st.markdown("**2. Hydrodynamic Size Shift**")
                    size_cols = st.columns(4)
                    size_cols[0].metric(f"{ch1_name} Median Size", f"{round(med_size_c1, 1)} nm")
                    size_cols[1].metric(f"{ch2_name} Median Size", f"{round(med_size_c2, 1)} nm")
                    size_cols[2].metric("Coloc. Median Size", f"{round(med_size_coloc, 1)} nm")
                    size_cols[3].metric("Size Status", size_score)
                    st.info(f"**Interpretation:** {size_exp}")
                    
                    # --- NEW: QC Data Export ---
                    qc_data = {
                        "Metric": [
                            "Pearson Correlation (r)", "R-squared (R²)", "Correlation Quality", "Correlation Interpretation",
                            f"{ch1_name} Median Area", f"{ch2_name} Median Area", "Colocalized Area", "Aggregation Status", "Aggregation Interpretation",
                            f"{ch1_name} Median Size (nm)", f"{ch2_name} Median Size (nm)", "Colocalized Median Size (nm)", "Size Status", "Size Interpretation"
                        ],
                        "Value": [
                            round(r_val, 3), round(r_sq, 3), score, exp,
                            round(med_area_c1, 1), round(med_area_c2, 1), round(med_area_coloc, 1), area_score, area_exp,
                            round(med_size_c1, 1), round(med_size_c2, 1), round(med_size_coloc, 1), size_score, size_exp
                        ]
                    }
                    
                    qc_csv = pd.DataFrame(qc_data).to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download QC Summary (CSV/Excel)",
                        data=qc_csv,
                        file_name=f"QC_Summary_{item['label'].replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                    st.markdown("---")
