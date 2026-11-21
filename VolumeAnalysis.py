import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

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

def ensure_folder(folder):
    """
    Ensure the target folder exists; create it if it doesn't.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

def load_and_process_data(file_path):
    """
    Load the CSV file and process the data:
      - Convert 'StartTime' to datetime (stored in a new column 'date')
      - Extract year and month
      - Calculate total Volume per year and per month
    """
    data = pd.read_csv(file_path)
    print("Columns in CSV:", data.columns)
    
    # Convert StartTime to datetime and extract year/month
    data['date'] = pd.to_datetime(data['StartTime'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month

    # Calculate total Volume per year and per month
    total_volume_year = data.groupby('year')['Volume'].sum()
    total_volume_month = data.groupby('month')['Volume'].sum()

    return data, total_volume_year, total_volume_month

def plot_price_volume_scatter(data, save_folder):
    """
    Create and save a scatter plot of average daily Price vs daily Volume for 2024.
    """
    # Filter data for the year 2024
    data_2024 = data[data['year'] == 2024].copy()
    
    # Create a new column that contains only the date (without time)
    data_2024['date_only'] = data_2024['date'].dt.date
    
    # Group by the date and aggregate: sum the Volume and average the Price
    daily_data = data_2024.groupby('date_only').agg({
        'Volume': 'sum',
        'Price': 'mean'
    }).reset_index()
    
    title = "Daily Average Price vs Daily Volume (2024)"
    fig = plt.figure(figsize=(10, 6))
    
    # Scatter plot: x-axis as average daily Price, y-axis as daily Volume
    plt.scatter(daily_data['Price'], daily_data['Volume'], alpha=0.5, c='blue', edgecolor='k')
    plt.xlabel('Average Daily Price (GBP/MWh)')
    plt.ylabel('Daily Volume (MWh)')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the figure without an on-plot title
    filename = os.path.join(save_folder, sanitize_filename(title) + ".png")
    fig.savefig(filename, dpi=300)
    plt.show()


def plot_total_volume_year(total_volume_year, save_folder):
    # Remove the unwanted year categories if they exist
    total_volume_year = total_volume_year.drop(labels=[2016, 2025], errors='ignore')
    
    title = "Total Volume vs Year"
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Plot the data with UK English color spelling
    total_volume_year.plot(kind='bar', color='orange', edgecolor='black', ax=ax)
    plt.xlabel('Year')
    plt.ylabel('Total Volume (Million MWh)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Format the y-axis ticks to show values in millions
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{x/1e6:,.0f}'))
    
    plt.tight_layout()
    
    filename = os.path.join(save_folder, sanitize_filename(title) + ".png")
    fig.savefig(filename, dpi=300)
    plt.show()

def main():
    file_path = r"C:\Users\joydo\Documents\RP3\PythonCode\2016-2025 - FullDataSet.csv"
    save_folder = r"C:\Users\joydo\Documents\RP3\PythonCode\Images"
    
    ensure_folder(save_folder)
    
    data, total_volume_year, total_volume_month = load_and_process_data(file_path)
    
    plot_price_volume_scatter(data, save_folder)
    plot_total_volume_year(total_volume_year, save_folder)

if __name__ == "__main__":
    main()
