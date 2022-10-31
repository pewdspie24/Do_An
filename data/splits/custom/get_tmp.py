with open("data/splits/custom/all.txt", "r") as f:
    all_file = f.read().splitlines()

for line in all_file:
    if " 5" in line:
        with open("data/splits/custom/5.txt", "a") as f:
            f.write(line+"\n")