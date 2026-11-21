import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Configure global font sizes before creating any figures
plt.rcParams.update({
    'font.size': 15,          # Global font size
    'axes.labelsize': 25,     # Axis label font size
    'xtick.labelsize': 20,    # X-axis tick label font size
    'ytick.labelsize': 20,    # Y-axis tick label font size
    'legend.fontsize': 15     # Legend font size
})

def sanitize_filename(filename):
    """
    Replace characters that are not allowed in Windows filenames.
    """
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def plot_electricity_slots_separate(df):
    # Define the folder where images will be saved
    save_folder = r"C:\Users\joydo\Documents\RP3\PythonCode\Images"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # ----------------------------------------------------------------------
    # 1) Ensure we have the necessary columns
    # ----------------------------------------------------------------------
    if not pd.api.types.is_datetime64_any_dtype(df['StartTime']):
        df['StartTime'] = pd.to_datetime(df['StartTime'], errors='coerce')

    if 'hour' not in df.columns:
        df['hour'] = df['StartTime'].dt.hour

    if 'dayOfWeek' not in df.columns:
        df['dayOfWeek'] = df['StartTime'].dt.day_name()

    if 'month' not in df.columns:
        df['month'] = df['StartTime'].dt.month_name()

    # Define order for day of the week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df['dayOfWeek'] = pd.Categorical(df['dayOfWeek'], categories=day_order, ordered=True)

    # ----------------------------------------------------------------------
    # 2) Figure 1: Price vs Month (Violin Plot)
    # ----------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    title_fig3 = "Price vs. Month"

    # Define the correct order for months
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    # Convert month column to categorical with specified order
    df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)

    sns.violinplot(
        x="month",
        y="Price",
        data=df,
        order=month_order,
        palette="coolwarm",
        inner="quartile",
        ax=ax3
    )

    ax3.set_xlabel("Month")
    ax3.set_ylabel("Price (GBP/MWh)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    fig3.savefig(os.path.join(save_folder, sanitize_filename(title_fig3) + ".png"), dpi=300)
    plt.show()  # Show the figure

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\joydo\Documents\RP3\PythonCode\2016-2025 - FullDataSet.csv")
    plot_electricity_slots_separate(df)
