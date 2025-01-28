import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to load data from the Excel file
def load_data(file_path, sheet_name):
    # A4. Please refer to the data present in “IRCTC Stock Price” data sheet of the above excel file. 
    # Do the following after loading the data to your programming platform.
    return pd.read_excel(file_path, sheet_name=sheet_name)

# Function to calculate the mean and variance of price data
def calculate_mean_variance(price_data):
    # • Calculate the mean and variance of the Price data present in column D.
    # (Suggestion: if you use Python, you may use statistics.mean() & statistics.variance() methods).
    population_mean = np.mean(price_data)
    population_variance = np.var(price_data)
    return population_mean, population_variance

# Function to calculate sample mean for Wednesdays and compare with population mean
def calculate_wednesday_mean(data):
    # • Select the price data for all Wednesdays and calculate the sample mean. Compare the mean
    # with the population mean and note your observations.
    wednesday_data = data[data["Day"] == "Wed"]
    wednesday_price = wednesday_data["Price"]
    wednesday_mean = np.mean(wednesday_price)
    return wednesday_mean

# Function to calculate sample mean for April and compare with population mean
def calculate_april_mean(data):
    # • Select the price data for the month of Apr and calculate the sample mean. Compare the
    # mean with the population mean and note your observations.
    april_data = data[data["Month"] == "Apr"]
    april_price = april_data["Price"]
    april_mean = np.mean(april_price)
    return april_mean

# Function to calculate the probability of making a loss over the stock
def calculate_probability_of_loss(data):
    # • From the Chg% (available in column I) find the probability of making a loss over the stock.
    # (Suggestion: use lambda function to find negative values)
    chg_percent = data["Chg%"]
    loss_days = chg_percent[chg_percent.apply(lambda x: x < 0)].count()
    total_days = chg_percent.count()
    probability_of_loss = (loss_days / total_days) * 100
    return probability_of_loss

# Function to calculate the probability of making a profit on Wednesday
def calculate_profit_on_wednesday(wednesday_data):
    # • Calculate the probability of making a profit on Wednesday.
    wed_chg_percent = wednesday_data["Chg%"]
    wed_profit_days = wed_chg_percent[wed_chg_percent.apply(lambda x: x > 0)].count()
    wed_total_days = wed_chg_percent.count()
    wed_profit_probability = (wed_profit_days / wed_total_days) * 100
    return wed_profit_probability

# Function to calculate the conditional probability of making profit, given that today is Wednesday
def calculate_conditional_profit_probability(wed_profit_probability, wed_total_days, total_days):
    # • Calculate the conditional probability of making profit, given that today is Wednesday.
    wed_probability = wed_total_days / total_days
    conditional_profit_probability = (wed_profit_probability / wed_total_days) * 100
    return conditional_profit_probability

# Function to create a scatter plot of Chg% data against the day of the week
def plot_chg_percent_vs_day_of_week(data):
    # • Make a scatter plot of Chg% data against the day of the week
    days = data["Day"]
    day_mapping = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    days_numeric = days.map(day_mapping)
    chg_percent_100 = data["Chg%"] * 100
    plt.figure(figsize=(10, 6))
    plt.scatter(days_numeric, chg_percent_100, alpha=0.6)
    plt.xticks(ticks=list(day_mapping.values()), labels=list(day_mapping.keys()))
    plt.xlabel("Day of the Week")
    plt.ylabel("Chg%")
    plt.title("Scatter Plot of Chg% Against Day of the Week")
    plt.grid(True)
    plt.show()

# Main function to orchestrate the entire process
def main(file_path, sheet_name):
    # Load data
    data = load_data(file_path, sheet_name)

    # Calculate mean and variance of price data
    population_mean, population_variance = calculate_mean_variance(data["Price"])
    print(f"The population mean price is: {population_mean:.2f}\n")
    print(f"The population variance price is: {population_variance:.2f}\n")

    # Calculate mean for Wednesdays
    wednesday_mean = calculate_wednesday_mean(data)
    print(f"\nThe mean price on Wednesdays is: {wednesday_mean:.2f}")

    # Calculate mean for April
    april_mean = calculate_april_mean(data)
    print(f"\nThe mean price in April is: {april_mean:.2f}")

    # Calculate probability of making a loss
    probability_of_loss = calculate_probability_of_loss(data)
    print(f"\nThe probability of making a loss over this stock is: {probability_of_loss:.3f}%")

    # Calculate probability of making a profit on Wednesday
    wednesday_data = data[data["Day"] == "Wed"]
    wed_profit_probability = calculate_profit_on_wednesday(wednesday_data)
    print(f"\nThe probability of making a profit over this stock on Wednesday is: {wed_profit_probability:.3f}%")

    # Calculate conditional probability of making profit on Wednesday
    wed_total_days = wednesday_data["Chg%"].count()
    total_days = data["Chg%"].count()
    conditional_profit_probability = calculate_conditional_profit_probability(wed_profit_probability, wed_total_days, total_days)
    print(f"\nThe conditional probability of making a profit given that today is Wednesday is: {conditional_profit_probability:.3f}%")

    # Plot Chg% against the day of the week
    plot_chg_percent_vs_day_of_week(data)

# If this script is being executed directly, call the main function
if __name__ == "__main__":
    file_path = "lab_session_data.xlsx"  # Path to the Excel file
    sheet_name = "IRCTC Stock Price"  # Name of the sheet in the Excel file
    main(file_path, sheet_name)