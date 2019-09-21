import raw_dataset.freiburg_driving as flying

Flying = flying.split_dataset([1, 1, 0])

print(len(Flying.full_dataset['training_set']))
print(len(Flying.full_dataset['test_set']))