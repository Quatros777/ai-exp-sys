import random
import matplotlib.pyplot as plt
import numpy as np

# Задаем случайные данные о количестве продаж по 10 товарам в течение 12 месяцев
random.seed(42)  # для воспроизводимости случайных чисел
num_products = 10
num_months = 12

sales_data = {f'Product {i}': [random.randint(50, 200) for _ in range(num_months)] for i in range(1, num_products + 1)}

# Построим график продаж товаров
months = list(range(1, num_months + 1))
for product, sales in sales_data.items():
    plt.plot(months, sales, label=product)

plt.title('График продаж товаров по месяцам')
plt.xlabel('Месяц')
plt.ylabel('Количество продаж')
plt.legend()
plt.show()


forecast_data = {}
for product, sales in sales_data.items():
    # Применяем метод скользящего среднего
    forecast = np.mean(sales[-3:])
 количество месяцев для анализа
    forecast_data[product] = forecast

color_ranges = {
    'high': 150,
    'medium': 100,
}

# Создадим список цветов для каждого продукта в зависимости от прогноза
colors = ['green' if forecast > color_ranges['high'] else 'yellow' if forecast > color_ranges['medium'] else 'red' for forecast in forecast_data.values()]

plt.barh(list(forecast_data.keys()), list(forecast_data.values()), color=colors)
plt.title('Прогноз продаж на 13-й месяц')
plt.xlabel('Прогноз продаж')
plt.ylabel('Товар')
plt.show()
