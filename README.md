# multi-object-tracking
Projekt zaliczeniowy z przedmiotu Sztuczna Inteligencja w Robotyce

Autor: Maciej Piotr Krupka

## Modele grafowy
Jako model grafowy został wykorzystany graf dwudzielny (ang. **Bipartite graph**)

<img width="1293" alt="bipartite_graph" src="https://github.com/macnack/multi-object-tracking/assets/59151986/83bdc6aa-ab88-42ae-82e4-9a430190e897">
[źródło](https://youtu.be/8uTXar8AWOw?t=839)

Jak na obrazie powyzej kolumny odpowiadają obiektą z obecnej klatki, a wiersze obiektą z poprzedniej klatki. Wartość odpowiada mierze prawdopodobieństwa dopasowania miedzy obiektami. W ramach projektu został dodany dodatkowy wiersz, który odpowiada za brak wystąpienia obiektu w klatce poprzedniej.

### Optymalizacja grafu
W celu rozwiazania przypisania, czyli zwrocenia najlepszego wyniku został wykorzystana metoda węgierska (ang.**Hungarian algorithm**) wykorzystując implementacje biblioteki [Scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)

## Metryki

### Korelacja histogramu
Obliczane jest podobieństwo między histogramami obiektów na podstawie współczynnika korelacji. W tym przypadku, korzystamy z dwuwymiarowych histogramów, które uwzględniają zarówno odcienie barw (Hue) jak i nasycenie (Saturation). Dzięki temu metoda działa nawet gdy zmienia się ich położenie lub kształt.

### Indeks Jaccarda
Współczynnik Jaccarda mierzy podobieństwo między dwoma obiektami i jest zdefiniowany jako iloraz mocy części wspólnej zbiorów i mocy sumy tych zbiorów. 

### Structural similarity
Funkcja oblicza wartość podobieństwa strukturalnego (SSIM) między dwoma obiektami. SSIM jest miarą podobieństwa strukturalnego między dwoma obrazami, uwzględniając zarówno informacje o jasności, kontraście, jak i strukturze.