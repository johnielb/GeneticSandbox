import sys


def knapsack(files):
    for i in range(1, len(files)):
        print("===", files[i], "===")
        file = open(files[i])
        header = file.readline().split(" ")
        n = int(header[0])
        capacity = int(header[1])
        print(n, "items, capacity", capacity)

        values = []
        weights = []
        for line in file:
            split = line.split(" ")
            values.append(int(split[0]))
            weights.append(int(split[1]))
        print(values)
        print(weights)
        assert len(values) == n
        assert len(weights) == n


if __name__ == '__main__':
    knapsack(sys.argv)
