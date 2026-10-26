# Projekt w CV i na rozmowie

## Opis do sekcji „Projekty”

**Customer Churn — PySpark, Databricks, MLflow**
Repozytorium: https://github.com/Benvenuteo/customer-churn-databricks-mlflow

- Zbudowanie pipeline'u przygotowania i walidacji ponad 505 tys. rekordów: warstwy bronze/silver, kwarantanna błędnych danych, kontrola schematu, duplikatów i zgodności liczby rekordów.
- Implementacja i porównanie modeli klasyfikacyjnych oraz regresyjnych w PySpark MLlib z walidacją krzyżową, przetwarzaniem dopasowanym wyłącznie do danych treningowych i niezależną oceną na drugim zbiorze.
- Rejestrowanie eksperymentów i kompletnych pipeline'ów w MLflow, wersjonowanie modeli, weryfikacja zgodności predykcji po zapisie i odczycie oraz predykcja wsadowa dla nowych rekordów.
- Diagnostyka różnic rozkładów danych i spadku ROC-AUC z 0,997 do 0,616; raportowanie kalibracji, jakości rankingu klientów oraz przedziałów ufności.

**Technologie:** Python, PySpark, Spark SQL / DataFrame API, MLlib, MLflow, Parquet, pytest, Docker, GitHub Actions. Integracja Databricks: notebooki, Delta i parametryzowane zadanie w pliku bundle.

## Co powiedzieć o Databricks

Punktem wyjścia był projekt studencki wykonany w Azure Databricks z MLflow i AutoML. Przebudowana wersja ma udokumentowane wyniki z lokalnego Sparka oraz przygotowane notebooki i konfigurację zadania Databricks. Nowego wykonania w chmurze ani nowego wyniku AutoML jeszcze nie potwierdzono. Takie rozróżnienie pozwala precyzyjnie opisać własną praktykę i stan repozytorium.

## Trzy minuty prezentacji

1. **Problem:** sprawdzam, czy model rozpoznający dostarczoną etykietę churn zachowuje jakość na drugim pliku; osobno estymuję zarejestrowane wydatki. Dane nie pozwalają określić horyzontu przyszłego churnu ani waluty.
2. **Dane:** 440 833 wiersze wejściowe w zbiorze rozwojowym, jeden całkowicie pusty wiersz w kwarantannie i 64 374 rekordy w pliku zewnętrznym. Identyfikator klienta nie trafia do cech modelu.
3. **Metoda:** podział rozwojowy 70/15/15, CV wyłącznie na próbce treningowej, końcowe dopasowanie na całym treningu i wybór progu na walidacji. Test wewnętrzny i plik zewnętrzny nie wybierają modelu.
4. **Wynik:** model GBT osiąga ROC-AUC 0,997 wewnętrznie i 0,616 zewnętrznie. Lift najlepszego decyla na drugim pliku wynosi 0,891, co nie uzasadnia wykorzystania tego rankingu do priorytetyzacji klientów.
5. **Wniosek:** pokazuję ograniczenie modelu, różnice między źródłami i odtwarzalny sposób jego oceny. Następny eksperyment wymaga wiarygodnego niezależnego źródła lub danych z czasem obserwacji; dalsze dostrajanie do tych samych etykiet zewnętrznych zatarłoby sens testu.

## Co pokazać w repozytorium

- [Wykonany notebook z wynikami](../notebooks/01_study_review.ipynb).
- [Raport HTML z osadzonymi wykresami](../reports/provided-v1/report/index.html) — pobierz i otwórz w przeglądarce.
- [Kontrakt danych i reguły jakości](../src/churn_study/data.py).
- [Rejestrację całego modelu i kontrolę zapisu/odczytu](../src/churn_study/tracking.py).
- [Testy](../tests), [automatyzację CI](../.github/workflows/ci.yml) oraz [instrukcję Databricks](DATABRICKS.md).

Jest to projekt portfolio i badanie porównawcze. Wyniki nie stanowią potwierdzenia wdrożenia produkcyjnego ani efektu biznesowego kampanii utrzymania klientów.
