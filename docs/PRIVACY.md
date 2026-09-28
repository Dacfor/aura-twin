# Dichiarazione e Architettura di Privacy-by-Design

## 1. Principi Fondamentali

Nelle missioni spaziali con equipaggio (NASA, ESA, ASI), la riservatezza psicologica e medica degli astronauti è tutelata da protocolli operativi stringenti. AURA-Face implementa un'architettura **Privacy-by-Design** certificabile:

1. **Zero Frame su Disco**: 
   - I frame catturati dalla videocamera risiedono **esclusivamente nella RAM volatile** per la frazione di secondo necessaria all'estrazione dei 478 landmark geometrici e delle 52 blendshape.
   - Nessun frame, ritaglio o immagine compressa (JPEG, PNG) viene mai scritto sul disco rigido o memorizzato in cache.
   - Nel codice sorgente di produzione (`src/aura_face/`), chiamate come `cv2.imwrite` o `cv2.imencode` sono assenti e verificate tramite test automatico CI (`tests/test_privacy.py`).

2. **Zero Cloud / Elaborazione Edge al 100%**:
   - Tutta l'inferenza di computer vision viene eseguita localmente sulla CPU/GPU del computer di bordo o dell'habitat.
   - Nessuna chiamata API di rete verso server esterni.
   - Funziona in totale isolamento offline, condizione obbligatoria per le stazioni lunari dove la latenza radio e i blackout orbitali impediscono lo streaming continuo verso la Terra.

3. **Telemetria Esclusivamente Numerica**:
   - I dati esportati e salvati nel database SQLite (`data/aura_face_telemetry.db`) consistono unicamente in vettori di numeri decimali (timestamp, EAR, PERCLOS, intensità AU da 0 a 1) e label di stato.
   - È matematicamente e fisicamente impossibile ricostruire l'immagine del volto dell'astronauta a partire dalla telemetria a 1 Hz memorizzata.

---

## 2. Test di Conformità nel Repository

Il repository include un test di audit statico continuo:
```bash
pytest tests/test_privacy.py -v
```
Questo test analizza programmaticamente tutti i file sorgente per verificare la conformità a questi vincoli.
