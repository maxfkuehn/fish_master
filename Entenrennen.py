


Entenanzahl = 8000
Verlierchance_einzel = 7999
Verlierchance_ergebnis = 1

for _ in range(150):
    Verlierchance_ergebnis *= Verlierchance_einzel / Entenanzahl
    Verlierchance_einzel -= 1
    Entenanzahl -= 1

print(f'Gewinnchance: ',(1-Verlierchance_ergebnis))