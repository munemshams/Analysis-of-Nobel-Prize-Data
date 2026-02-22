import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("nobel.csv")

# -------------------- 1. Basic Cleanup --------------------
df["decade"] = (df["year"] // 10) * 10

# -------------------- 2. Most common gender & birth country --------------------
top_gender = df["sex"].mode()[0]
top_country = df["birth_country"].mode()[0]

# -------------------- 3. Decade with highest ratio of US-born laureates --------------------
total_per_decade = df.groupby("decade")["full_name"].count()
usa_per_decade = df[df["birth_country"] == "United States of America"].groupby("decade")["full_name"].count()
ratio = (usa_per_decade / total_per_decade).fillna(0)

max_decade_usa = int(ratio.idxmax())

# -------------------- 4. Highest proportion of female winners (decade × category) --------------------
female_df = df[df["sex"] == "Female"]

female_counts = female_df.groupby(["decade", "category"])["full_name"].count()
all_counts = df.groupby(["decade", "category"])["full_name"].count()
female_prop = (female_counts / all_counts).fillna(0)

max_pair = female_prop.idxmax()
max_female_dict = {int(max_pair[0]): max_pair[1]}

# -------------------- 5. First woman Nobel laureate --------------------
first_female = df[df["sex"] == "Female"].sort_values("year").iloc[0]
first_woman_name = first_female["full_name"]
first_woman_category = first_female["category"]

# -------------------- 6. Repeat winners --------------------
repeat_list = (
    df["full_name"]
    .value_counts()
    .loc[lambda x: x > 1]
    .index
    .tolist()
)

# -------------------- Print results --------------------
print("top_gender:", top_gender)
print("top_country:", top_country)
print("max_decade_usa:", max_decade_usa)
print("max_female_dict:", max_female_dict)
print("first_woman_name:", first_woman_name)
print("first_woman_category:", first_woman_category)
print("repeat_list:", repeat_list)

# -------------------- Visualizations --------------------
sns.set_theme(style="whitegrid")

# Gender over decades
plt.figure(figsize=(10,6))
sns.countplot(data=df, x="decade", hue="sex", palette="Set2")
plt.title("Nobel Prizes by Gender Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("nobel_gender_over_time.png")
plt.show(block=False)
plt.pause(1)
plt.close()

# Top 15 birth countries
plt.figure(figsize=(10,6))
top_birth_countries = df["birth_country"].value_counts().head(15)
sns.barplot(x=top_birth_countries.values, y=top_birth_countries.index, palette="Blues_r")
plt.title("Top 15 Nobel Laureate Birth Countries")
plt.tight_layout()
plt.savefig("nobel_top_birth_countries.png")
plt.show(block=False)
plt.pause(1)
plt.close()

# Female proportion by category
female_category_prop = (
    (df[df["sex"] == "Female"].groupby("category")["full_name"].count() /
     df.groupby("category")["full_name"].count())
    .sort_values(ascending=False)
)

plt.figure(figsize=(10,6))
sns.barplot(x=female_category_prop.values, y=female_category_prop.index, palette="PuRd")
plt.title("Proportion of Female Laureates by Category")
plt.tight_layout()
plt.savefig("nobel_female_proportion_by_category.png")
plt.show(block=False)
plt.pause(1)
plt.close()

# US-born ratio over decades
plt.figure(figsize=(10,6))
plt.plot(ratio.index, ratio.values, marker="o", color="red")
plt.title("Ratio of US-born Nobel Laureates by Decade")
plt.xlabel("Decade")
plt.ylabel("US-born Ratio")
plt.grid(True)
plt.tight_layout()
plt.savefig("nobel_us_ratio_over_time.png")
plt.show(block=False)
plt.pause(1)
plt.close()

# Heatmap female proportion
female_table = female_prop.unstack().fillna(0)

plt.figure(figsize=(12,6))
sns.heatmap(female_table, annot=True, cmap="YlOrRd", fmt=".2f")
plt.title("Female Laureate Proportion — Heatmap (Decade × Category)")
plt.tight_layout()
plt.savefig("nobel_female_heatmap.png")
plt.show(block=False)
plt.pause(1)
plt.close()
