# Protocollo Sperimentale di Validazione

> Risposta al quesito del bando BEX2026 Challenge #3:  
> *"How would you verify it? Indicate how you could determine whether your solution actually works: which behavior or parameter you would observe, on whom and in what context."*

---

## 1. Disegno Sperimentale

- **Ambiente di Test**: Campagna in missione analoga di isolamento (es. **ESA CAVES** nelle grotte della Sardegna o **HI-SEAS** a Mauna Loa, Hawaii).
- **Campione**: $N = 12$ astronauti analoghi (o personale sanitario turnista in reparto di emergenza) monitorati per 14 giorni consecutivi.
- **Variabili Indipendenti**:
  1. Condizione di riposo basale (post-sonno ristoratore, $t=0$).
  2. Carico cognitivo crescente (aritmetica mentale complessa, test di Stroop, risoluzione anomalie di bordo).
  3. Deprivazione circadiana (fine turno di 12 ore nella notte biologica).

---

## 2. Parametri Osservati e Ground Truth (Misure di Controllo)

1. **AURA-Face Optical Telemetry**:
   - `PERCLOS`: percentuale di tempo in chiusure palpebrali lente su finestra mobile a 60 secondi.
   - `EAR (Eye Aspect Ratio)`: dinamica dell'apertura palpebrale e durata mediana del blink (ms).
   - `Frequenza Microsleep`: chiusure oculari involontarie con durata $\ge 500$ ms.
   - `AU4 (Brow Lowerer)`: indice di sforzo mentale sostenuto.

2. **Ground Truth Psicofisiologico Concomitante**:
   - **PVT-B (Brief Psychomotor Vigilance Task, 3 minuti)**: Gold standard NASA per la misurazione oggettiva dei tempi di reazione e dei *lapse* attentivi ($RT > 500$ ms).
   - **Karolinska Sleepiness Scale (KSS)**: Valutazione soggettiva dello stato di sonnolenza (scala standardizzata 1–9) somministrata a intervalli regolari.
   - **Wearable Cardiovascolare**: Misurazione indipendente di Heart Rate Variability (HRV - RMSSD) per quantificare la regolazione del sistema nervoso autonomo.

---

## 3. Criteri di Validazione Statistica e Benchmark

- **Ipotesi di Validazione**: Correlazione lineare di Pearson tra PERCLOS calcolato da AURA-Face e i lapse al PVT-B con coefficiente $r \ge 0.85$ ($p < 0.001$).
- **Benchmark Eseguibile in Repository**:
  ```bash
  python scripts/validate_protocol.py
  ```
- **Risultati Ottenuti nel Benchmark**:
  - Pearson $r(\text{PERCLOS}, \text{PVT Lapses}) = 0.9078$ (Ipotesi superata con successo: $r > 0.85$).
  - Pearson $r(\text{PERCLOS}, \text{Scala KSS}) = 0.9267$.
  - Accuratezza Diagnostica Complessiva: $88.9\%$.
  - Sensibilità (Recall nella rilevazione della sonnolenza): $95.5\%$.
  - Specificità: $78.6\%$.

Il file `docs/validation_report.json` raccoglie tutti i dati statistici grezzi e la matrice di confusione esportata.
