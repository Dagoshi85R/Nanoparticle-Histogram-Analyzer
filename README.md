Nanoparticle Histogram Analyzer
A powerful, web-based Streamlit application designed to generate publication-quality visualizations and advanced statistical analyses from raw .zip files exported by ZetaSphere (Particle Metrix ZetaView) NTA instruments.

It effortlessly handles multiplexed fluorescent channels, performs automated non-parametric statistical comparisons, detects heterogeneous sub-populations via machine learning (GMM), and evaluates fluorescent colocalization quality—all without writing a single line of code.

🌐 Access the Live Web App Here: [Insert your Streamlit Cloud Link Here]

✨ Key Features
Dual Rendering Engines: Choose between strict, mathematically rigid "ZetaSphere-style" anchored polygons for absolute analytical accuracy, or toggle on "Smooth KDE Curves" for aesthetically pleasing, publication-ready continuous distributions.

Region of Interest (ROI) Gating: Visually gate specific size ranges (e.g., 30 nm – 150 nm for EVs) to automatically calculate the exact particle count and percentage of the population falling within that specific Area Under the Curve (AUC).

Stable Peak Math: Histogram bin widths are fully adjustable for visual clarity, but the statistical "Mode" is secretly powered by a high-resolution Kernel Density Estimate (KDE) behind the scenes, ensuring your peak data is 100% stable and reproducible regardless of visual settings.

Intelligent Pooling & Multi-Sample Overlays: Drag and drop dozens of .zip files to compare biological replicates. Choose to view them independently, array them in Facet Grids or Ridgeline (Joyplot) plots, or pool replicates to calculate global medians and standard deviations.

Automated Cross-Stats: Instantly calculates All-vs-All pairwise statistics across active samples using Mann-Whitney U (Median shifts), Anderson-Darling (Shape/tail variations), and Earth Mover's Distance (Physical histogram divergence).

🛠️ The Analysis Modules
1. Multi-Sample Size Distribution
Visualizes the Hydrodynamic Diameter (nm) of your particles. Calculates absolute counts, Mean, Median, Mode, and Standard Deviation. Includes ROI Gating to instantly quantify specific sub-populations (e.g., separating small vesicles from large aggregates).

2. Zeta Potential & Bivariate Scatter
Analyzes surface charge distributions (mV). Includes advanced bivariate scatter plots mapping Size vs. Zeta Potential with marginal histograms/densities along the axes, allowing you to directly correlate physical size with charge profiles across multiple fluorescent channels.

3. Concentration Analysis
Generates customizable Bar Charts, Dot Plots (Strip), or Box Plots for absolute particle counts.

Dilution Factors: Manually input the dilution factor to calculate true physical Concentration (particles/mL).

Replicate Grouping: Assign the exact same "Label" to multiple files to automatically group them as biological replicates and generate Standard Deviation error bars.

4. Advanced Population Analysis (GMM)
Ideal for heavily heterogeneous samples.

Uses unsupervised machine learning (Gaussian Mixture Models) to automatically detect and mathematically separate underlying sub-populations.

Uses the Bayesian Information Criterion (BIC) to objectively score and determine the optimal number of populations.

Exports a comprehensive, multi-sheet .xlsx Excel report containing all global stats, sub-population summaries, BIC scores, and raw curve/histogram generation data.

5. Colocalization Quality Control
An advanced diagnostic module for dual-labeled samples. Evaluates the physical reality of the machine's colocalization calls by analyzing two critical metrics:

Dye Stoichiometry: Plots the Mean Intensity of Channel 1 vs Channel 2, calculating Pearson Correlation (r) and R-squared to determine if fluorophore binding is proportional or random.

Hydrodynamic Size Shift: Compares the size of single-labeled particles against dual-labeled particles to ensure the labeling process isn't artificially inducing sample aggregation. Outputs a downloadable QC CSV report.

📂 Expected Input Format
Simply drag and drop the raw .zip files generated directly by the ZetaSphere software. The app automatically unzips the file, locates the internal CSVs, and parses the metadata (Measurement Type, Channel, and Sample ID) directly from the filename.

Supported Channels: Scatter (488s), 405 nm, 488 nm, 520 nm, and 640 nm.

🚀 How to Run Locally
While the app is primarily designed to be accessed via the web browser for ease of use, you can easily run it locally on your own machine for offline processing:

Ensure Python 3.8+ is installed.

Clone this repository and navigate to the folder.

Install the required dependencies: pip install -r requirements.txt

Launch the Streamlit app: streamlit run Z_histograms1.0.py
