train_set[0]

dataframe.head()

sns.set_theme()

dataframe["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()

print(tokenize(dataset["train"][:2]))
print(tokenized_datasets["train"].column_names)