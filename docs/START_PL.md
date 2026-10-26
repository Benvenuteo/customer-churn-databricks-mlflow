# Uruchomienie projektu

Najprościej na Windows użyć Docker Desktop z włączonym WSL2. Rozpakuj projekt i umieść dwa oryginalne pliki CSV w `data/raw/`. W terminalu przejdź do katalogu z `docker-compose.yml`.

W PowerShell uruchamiaj poniższe polecenia jako pojedyncze wiersze:

```powershell
docker compose build study
docker compose run --rm study churn run --training data/raw/customer_churn_dataset-training-master.csv --external data/raw/customer_churn_dataset-testing-master.csv --output outputs/moj-run --run-id moj-run
```

Otwórz `outputs/moj-run/report/index.html`. Raport zawiera wykresy wewnątrz HTML, więc nie wymaga osobnego kopiowania obrazów.

Do szybkiej próby bez oryginalnych danych:

```powershell
docker compose run --rm study churn demo
docker compose run --rm study churn run --training data/demo/development.csv --external data/demo/external.csv --output outputs/demo-run --run-id demo-run --data-kind synthetic --profile smoke
```

Każde nowe uruchomienie przygotowania wymaga nowego katalogu wyjściowego. Jeśli przygotowanie zakończyło się poprawnie, a trening nie był jeszcze wykonany, można go rozpocząć osobno:

```powershell
docker compose run --rm study churn train --output outputs/moj-run
```

Po częściowym błędzie treningu zachowaj `bronze`, `silver`, `quarantine`, `manifest.json`, `quality.json` i `shift_audit.json`. Przed ponowieniem konkretnego zadania usuń tylko jego niekompletne `models/<task>`, `predictions/<task>` i `<task>.json`; dla zakończonego zadania wybierz nowe badanie zamiast go nadpisywać. Nie zmieniaj wyboru modelu na podstawie wyników testu zewnętrznego.

Java służy do uruchomienia silnika Spark. Kod projektu to Python/PySpark i SQL. W kontenerze Java jest już zainstalowana. Alternatywnie użyj lokalnego Pythona 3.11 i Javy 17 zgodnie z README; WSL2 ogranicza typowe problemy lokalnego Hadoop/Spark na Windows.

Gotowe modele demonstracyjne znajdują się w paczce w `examples/provided-v1/models`. Przy natywnym uruchomieniu można sprawdzić predykcję:

```powershell
churn score --input data/raw/customer_churn_dataset-testing-master.csv --model examples/provided-v1/models/classification --task classification --output outputs/predykcje
```

Model osiąga słaby wynik przenoszenia na drugi plik. Projekt pokazuje ten wynik i jego diagnostykę; nie przedstawia modelu jako sprawdzonego systemu utrzymania klientów.
