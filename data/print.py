import pickle

with open("genetic-gm", "rb") as f:
    matrices = pickle.load(f)

with open("genetic-mh", "rb") as f:
    mHeights = pickle.load(f)

for key, matrix in matrices.items():
    print("Key:", key)
    print("Matrix:")
    for row in matrix:
        print(row)
    print()

for key, mHeight in mHeights.items():
    print(f"{key}: mHeight = {mHeight:.04f}")

print("Done")