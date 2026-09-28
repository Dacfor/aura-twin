# AURA-Face: Dossier Strategico di Progetto (ASI BEX2026)

**Challenge #3: "Artemis Community. Designing Life on the Moon Between Science and the Humanities"**  
*Area 1: Space Psychology, Wellbeing & Relationships*  
*Area 2: Behavioral Design & Social Habitats*

---

## 1. Executive Summary

La permanenza prolungata sul suolo lunare nelle missioni del programma Artemis richiede un cambio di paradigma: la sopravvivenza non dipende unicamente dai sistemi ingegneristici di supporto vitale (ECLSS), ma dalla capacità di garantire la salute psicologica, la vigilanza operativa e la coesione sociale dell'equipaggio in un contesto estremo.

Sulla Luna, gli astronauti affrontano:
- **Totale isolamento e confinamento** in volumi abitativi ristretti.
- **Assenza di ciclo naturale giorno-notte** (una notte lunare dura 14 giorni terrestri), con grave compromissione dei ritmi circadiani e del sonno.
- **Latenza di comunicazione asimmetrica** verso la Terra (1.3 secondi minimi di ritardo luce per tratta, blackout orbitali), che impone una totale autonomia decisionale a bordo.

**AURA-Face** è il sensore ottico edge del Digital Twin multimodale per l'habitat lunare: misura in modo continuo e non invasivo il **comportamento facciale oggettivo** (Action Units FACS e dinamica oculare: EAR, PERCLOS, microsleep, sbadigli e carico cognitivo). Elabora tutto in locale (zero immagini salvate, zero cloud), si calibra sull'astronauta in meno di 25 secondi e produce una telemetria numerica a 1 Hz che alimenta l'ecosistema intelligente dell'habitat (luci biodinamiche, ripianificazione turni, supporto psicologico).

---

## 2. Risposta ai 4 Criteri di Valutazione della Giuria BEX2026

### Criterio 1: L'Approccio Multidisciplinare (Tecnologia Spaziale + Scienze Umane)
Il progetto realizza una sintesi autentica tra ingegneria informatica di frontiera e scienze umane e psicofisiologiche:
- **Fondamento Psicofisiologico**: Utilizza la metrica **PERCLOS** (Wierwille et al., 1994; Dinges & Grace, 1998) e il **Facial Action Coding System (FACS)** di Ekman & Friesen (1978), combinati con il modello prototipale EMFACS (Friesen & Ekman, 1984).
- **Onestà Scientifica e Rispetto della Letteratura Critica**: In linea con la ricerca di Lisa Feldman Barrett et al. (2019, *Emotional Expressions Reconsidered*), AURA-Face non pretende di "leggere la mente o le emozioni universali", ma dichiara e misura configurazioni muscolari e oculari oggettive correlate a vigilanza, carico cognitivo e fatica.
- **Edge Computing & Computer Vision 3D**: 478 landmark 3D e 52 blendshapes estratti in memoria volatile tramite MediaPipe Face Landmarker, con calibrazione statistica individuale.

### Criterio 2: Impatto Bidirezionale Spazio-Terra (Two-Way Impact)

#### Dallo Spazio alla Terra (Space for Society)
Le stesse vulnerabilità psicofisiologiche dell'astronauta lunare colpiscono categorie critiche sul pianeta Terra:
- **Personale Medico e Ospedaliero di Terapia Intensiva (ICU)**: I turni notturni massacranti inducono colpi di sonno (microsleep) e affaticamento decisionale. AURA-Face offre una sentinella non a contatto che non richiede elettrodi fastidiosi.
- **Sale di Controllo e Trasporti ad Alto Rischio**: Operatori ferroviari ad alta velocità, piloti di linea, controllori di volo e conducenti industriali.
- **Basi Remote e di Ricerca Estrema**: Personale isolato nelle stazioni antartiche (Concordia Station) e piattaforme offshore.

#### Dalla Terra allo Spazio (Society for Space)
- Integrazione delle metodologie terrestri di medicina del lavoro, del Psychomotor Vigilance Task (PVT-B) e della scala Karolinska (KSS) all'interno dell'interfaccia habitat Artemis.

### Criterio 3: Concept & User Journey (L'Esperienza Quotidiana dell'Astronauta)

1. **Inizio Turno (Wake-up & Workstation Check-in - 08:00 Lunar Time)**:
   - L'astronauta siede alla console operativa. Il sistema riconosce la presenza ed esegue una micro-calibrazione di 20 secondi in sottofondo, calibrando l'apertura oculare basale.
2. **Attività Operativa EVA / Controllo Sistemi (14:00)**:
   - Durante una procedura ad alto stress cognitivo, l'AU4 (corrugatore del sopracciglio) sale sopra 0.40 senza sorriso per oltre 3 secondi. Il cockpit HUD registra l'evento `COGNITIVE_LOAD_ONSET`. Il Digital Twin adatta la densità delle informazioni a schermo riducendo gli alert non prioritari.
3. **Fase di Fatica Circadiana (22:00 - Fine Turno nella Notte Lunare)**:
   - Dopo 14 giorni di buio esterno, la qualità del sonno cala. Gli ammiccamenti diventano lenti (>400 ms) e il PERCLOS raggiunge il 12%. Lo stato passa a `MODERATE`.
   - Se compare un microsleep da 650 ms, lo stato commuta istantaneamente in `CRITICAL`. Il cockpit HUD attiva un segnale visivo ambra/rosso e notifica il Digital Twin dell'habitat.
4. **Attuazione Sistemica (Il Ruolo del Digital Twin dell'Habitat)**:
   - AURA-Face non prende decisioni invasive; invia la telemetria a 1 Hz al sistema centrale, che modula le **luci biodinamiche circadiane** (aumento temperatura di colore verso blu a 480nm per bloccare la melatonina) e suggerisce una micro-pausa o l'avvicendamento con un compagno di equipaggio.

### Criterio 4: Come Validarlo Sperimentalmente? ("How would you verify it?")

Il disegno di validazione segue il protocollo del programma NASA OCR (*Optical Computer Recognition of Stress, Affect and Fatigue during Performance in Spaceflight*, Dinges & Metaxas, NSBRI):
1. **Campagna in Ambiente Analogo**:
   - Test condotto durante missioni analoghe ufficiali (ESA CAVES nelle grotte sarde o HI-SEAS alle Hawaii) su un equipaggio di $N=12$ analog astronauts in isolamento prolungato.
2. **Ground Truth Concomitante**:
   - **PVT-B (Brief Psychomotor Vigilance Task)**: test validato di 3 minuti per la misurazione oggettiva di tempi di reazione e cali di attenzione (lapse > 500 ms).
   - **Karolinska Sleepiness Scale (KSS)**: rating soggettivo validato (1-9) ogni 30 minuti.
   - **Wearable Fisiologico Autonomo**: monitoraggio concomitante di Heart Rate Variability (HRV RMSSD).
3. **Criteri di Successo Quantitativi**:
   - Correlazione di Pearson $r \ge 0.85$ tra PERCLOS e frequenza di lapse al PVT-B (il nostro benchmark simulato ha dimostrato $r = 0.9078$).
   - Accuratezza diagnostica $> 88\%$ nella classificazione degli stati di sonnolenza.
   - Matrice di confusione pubblica inclusa nel repository (`docs/validation_report.json`).

---

## 3. Conformità Etica e Privacy by Design

- **Zero Frame su Disco**: Nel codice sorgente (`src/aura_face/`) non esiste alcuna funzione di salvataggio immagini (`cv2.imwrite`, `cv2.imencode`). I frame video risiedono esclusivamente nella RAM volatile del frame grabber per il calcolo geometrico e vengono immediatamente sovrascritti.
- **Telemetria Esclusivamente Numerica**: Il database SQLite locale (`data/aura_face_telemetry.db`) memorizza unicamente numeri in virgola mobile (EAR, PERCLOS, gradi di orientamento testa, intensità AU) e label di stato.
- **Nessuna Trasmittanza Biometrica**: Rispetta gli standard etici e la riservatezza medica NASA/ESA per gli equipaggi spaziali.
