import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# ========== 1. ЗАГРУЗКА ДАННЫХ ==========
df = pd.read_csv(r'C:\Users\hikim\Downloads\Social_media_impact_on_life.csv')

# ========== 1.1. ПЕРВИЧНЫЙ ОСМОТР ==========
print("=== 1.1. Первичный осмотр ===")
print(df.head())
print(f"\nСтрок: {df.shape[0]}, Столбцов: {df.shape[1]}")
df.info()
print(f"\nПропусков: {df.isnull().sum().sum()}")
print(f"Дубликатов: {df.duplicated().sum()}")

# Категориальные столбцы
cat_cols = ['Gender', 'Academic_Level', 'Country', 'Most_Used_Platform',
            'Affects_Academic_Performance', 'Overall_Impact']
for col in cat_cols:
    print(f"{col}: {df[col].unique()[:10]}")

# ========== 1.2. РАСПРЕДЕЛЕНИЕ OVERALL IMPACT ==========
print("\n=== 1.2. Распределение Overall Impact ===")
impact_counts = df['Overall_Impact'].value_counts()
impact_percent = df['Overall_Impact'].value_counts(normalize=True) * 100
print(impact_counts)
print(impact_percent.round(1))

# График
impact_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Распределение студентов по оценке влияния соцсетей')
plt.xlabel('Оценка влияния')
plt.ylabel('Количество студентов')
plt.xticks(rotation=0)
plt.show()

# ========== 1.3. СРЕДНЕЕ ВРЕМЯ В СОЦСЕТЯХ ==========
print("\n=== 1.3. Среднее время в соцсетях ===")
print(f"Среднее: {df['Avg_Daily_Usage_Hours'].mean():.2f} ч")
print(f"Медиана: {df['Avg_Daily_Usage_Hours'].median():.2f} ч")
print(f"Максимум: {df['Avg_Daily_Usage_Hours'].max():.2f} ч")

print("\nСреднее по группам Overall_Impact:")
print(df.groupby('Overall_Impact')['Avg_Daily_Usage_Hours'].mean().round(2))

# ========== 1.4. ВЛИЯНИЕ НА УСПЕВАЕМОСТЬ ==========
print("\n=== 1.4. Влияние на успеваемость ===")
yes_count = len(df[df['Affects_Academic_Performance'] == 'Yes'])
yes_percent = (yes_count / len(df)) * 100
print(f"Процент ответивших Yes: {yes_percent:.1f}%")

yes_scores = df[df['Affects_Academic_Performance'] == 'Yes']['Mental_Health_Score']
no_scores = df[df['Affects_Academic_Performance'] == 'No']['Mental_Health_Score']
print(f"Группа Yes: средний Mental_Health_Score = {yes_scores.mean():.1f}")
print(f"Группа No:  средний Mental_Health_Score = {no_scores.mean():.1f}")

# ========== 2.1. КОРРЕЛЯЦИЯ ==========
print("\n=== 2.1. Корреляционный анализ ===")
numeric_cols = ['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score']
corr_matrix = df[numeric_cols].corr()
print(corr_matrix.round(3))

# Тепловая карта
plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
plt.title('Корреляция между показателями')
plt.show()

# ========== 2.2. СРАВНЕНИЕ ПЛАТФОРМ ==========
print("\n=== 2.2. Сравнение платформ ===")
def negative_percent(x):
    return (x == 'Negative').mean() * 100

platform_stats = df.groupby('Most_Used_Platform').agg(
    avg_usage=('Avg_Daily_Usage_Hours', 'mean'),
    avg_mental=('Mental_Health_Score', 'mean'),
    negative_pct=('Overall_Impact', negative_percent)
).round(1).sort_values('avg_usage', ascending=False)
print(platform_stats)

# График: среднее время по платформам
plt.figure(figsize=(12,6))
platform_stats['avg_usage'].plot(kind='bar', color='skyblue')
plt.title('Среднее время в соцсетях по платформам')
plt.xlabel('Платформа')
plt.ylabel('Среднее время (часы)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ========== 2.3. АНАЛИЗ ПО ДЕМОГРАФИИ ==========
print("\n=== 2.3. Анализ по полу ===")
gender_stats = df.groupby('Gender').agg(
    avg_usage=('Avg_Daily_Usage_Hours', 'mean'),
    avg_mental=('Mental_Health_Score', 'mean'),
    negative_pct=('Overall_Impact', negative_percent)
).round(1).reset_index()
print(gender_stats)

print("\n=== 2.3. Анализ по уровню образования ===")
education_stats = df.groupby('Academic_Level').agg(
    avg_usage=('Avg_Daily_Usage_Hours', 'mean'),
    avg_mental=('Mental_Health_Score', 'mean'),
    negative_pct=('Overall_Impact', negative_percent)
).round(1).reset_index()
print(education_stats)

# ========== 2.4. ПРОВЕРКА ГИПОТЕЗЫ (T-TEST) ==========
print("\n=== 2.4. T-Test: Mental Health Score по группам Yes/No ===")
t_stat, p_value = ttest_ind(yes_scores, no_scores)
print(f"p-value = {p_value:.4f}")
if p_value < 0.05:
    print("Разница статистически значима (p < 0.05)")
else:
    print("Разница НЕ статистически значима (p >= 0.05)")

# ========== 3. ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ ДЛЯ ДАШБОРДА ==========
print("\n=== 3. Дополнительные визуализации ===")
# Ящик с усами
plt.figure(figsize=(8,6))
df.boxplot(column='Mental_Health_Score', by='Overall_Impact', grid=False)
plt.title('Распределение Mental Health Score по группам влияния')
plt.suptitle('')
plt.xlabel('Оценка влияния')
plt.ylabel('Mental Health Score')
plt.show()

# ========== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ==========
gender_stats.to_csv('gender_stats.csv', index=False)
education_stats.to_csv('education_stats.csv', index=False)
platform_stats.to_csv('platform_stats.csv')
corr_matrix.to_csv('correlation_matrix.csv')
print("\nРезультаты сохранены в CSV-файлы: gender_stats.csv, education_stats.csv, platform_stats.csv, correlation_matrix.csv")