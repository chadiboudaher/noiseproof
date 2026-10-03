from noiseproof.data.grid import GRIDDataset


dataset = GRIDDataset(
    root_dir="data/raw/grid"
)

print("Number of samples:", len(dataset))
print(dataset[0])