# Fondamenti Scientifici di AURA-Face

AURA-Face è costruito rigorosamente sulla letteratura scientifica della medicina spaziale e della psicofisiologia. 
Questo documento riassume le basi teoriche ed empiriche per le slide e il Q&A con la giuria tecnica.

---

## 1. Riferimenti Chiave

1. **Dinges, D. F. & Metaxas, D. et al. (2008–2012)**  
   *Optical Computer Recognition of Stress, Affect and Fatigue during Performance in Spaceflight*, NSBRI/NASA Taskbook.  
   - Il precedente diretto del nostro sistema: programma NASA per il riconoscimento ottico non a contatto di stress, fatica e affetto durante il volo spaziale.  
   - Validato nell'analogo sottomarino NEEMO contro PVT, EEG, ECG e cortisolo salivare.  
   - Beneficiari terrestri dichiarati: piloti d'aereo, conducenti industriali, operatori di sale di controllo.

2. **Wierwille, W. W. et al. (1994)**  
   *Evaluation of Driver Drowsiness by Trained Raters*.  
   - Definizione canonica di PERCLOS: percentuale di tempo su un intervallo di 1 minuto in cui le palpebre rimangono chiuse ad almeno l'80%, contando specificamente le **chiusure lente ("droops")** e NON gli ammiccamenti fisiologici rapidi.

3. **Dinges, D. F. & Grace, R. (1998)**  
   *PERCLOS: A Valid Psychophysiological Measure of Alertness as Assessed by Psychomotor Vigilance*.  
   - Validazione psicofisiologica formale: dimostrazione che PERCLOS è il correlato biologico più accurato dei cali di attenzione e dei lapse psicomotori.

4. **Schleicher, R., Galley, N., Briest, S. & Galley, L. (2008)**  
   *Blinks and Saccades as Indicators of Fatigue in Computerized Tasks*, British Journal of Ophthalmology.  
   - Dimostra che la **durata del blink** e la **velocità di riapertura** sono gli indicatori più stabili della sonnolenza profonda. La frequenza dei blink aumenta solo nelle prime fasi di stanchezza leggera, per poi crollare durante il sovraccarico cognitivo.

5. **Ekman, P. & Friesen, W. V. (1978)**  
   *Facial Action Coding System (FACS)*.  
   - Scomposizione anatomica dei movimenti facciali in Action Units (AU) indipendenti.

6. **Friesen, W. V. & Ekman, P. (1984)**  
   *EMFACS (Emotional Facial Action Coding System)*.  
   - Combinazioni prototipali empiriche per stati affettivi discreti (godimento genuino AU6+AU12, rabbia/stress AU4+AU6).

7. **Barrett, L. F., Adolphs, R., Marsella, S., Martinez, A. M. & Pollak, S. D. (2019)**  
   *Emotional Expressions Reconsidered: Challenges to Inferring Emotion From Human Facial Movements*, Psychological Science in the Public Interest.  
   - **La nostra difesa scientifica principale al Q&A**: dimostra che le configurazioni facciali non corrispondono in modo universale e decontestualizzato a categorie emotive discrete interiori. Per questa ragione, AURA-Face effettua dichiarazioni rigorosamente **comportamentali e psicofisiologiche** (carico cognitivo, ammiccamenti lenti, microsleep), non di "lettura della mente".

---

## 2. Tabella Azioni Muscolari (FACS) e Correlati

| Action Unit FACS | Blendshape MediaPipe | Correlato Psicofisiologico | Utilizzo in AURA-Face |
|---|---|---|---|
| **AU4 Brow Lowerer** | `browDownLeft/Right` | Sforzo mentale, corrugamento, frustrazione, concentrazione intensa | Indicatore di carico cognitivo sostenuto |
| **AU6 Cheek Raiser** | `eyeSquintLeft/Right` | Sollevamento zigomatico, componente perioculare Duchenne | Distinzione sorriso autentico vs sociale |
| **AU12 Lip Corner Puller** | `mouthSmileLeft/Right` | Valenza positiva | Rilevamento engagement positivo / sollievo |
| **AU26/27 Jaw Drop** | `jawOpen` | Caduta mandibolare involontaria | Rilevamento sbadigli prolungati (>2 s) |
| **Oculare Bilaterale** | Landmark Mesh 33-160-158-133-153-144 / 362-385-387-263-373-380 | Apertura rima palpebrale | Calcolo EAR, classificazione blink / droop / microsleep e PERCLOS |

---

## 3. Modello Dimensionale dell'Affetto (Russell 1980; Barrett 2019)

Invece di affidarsi a classificatori ingenui a "6 emozioni di base" (che Barrett et al. 2019 dimostrano fallire drasticamente in ambienti operativi ecologici privi di contesto semantico), AURA-Face proietta l'attività mimica ed oculare in uno spazio ortogonale continuo:

1. **Valenza Edonica ($V \in [-1.0, +1.0]$)**:
   $$V = \text{AU12}_{\text{norm}} - \max(0.8 \cdot \text{AU4}_{\text{norm}},\, \text{AU15}_{\text{norm}},\, \text{AU9}_{\text{norm}})$$
   - Guidata positivamente dal muscolo grande zigomatico ($\text{AU12}$).
   - Guidata negativamente dal corrugatore sopracciliare ($\text{AU4}$), depressore dell'angolo della bocca ($\text{AU15}$) ed elevatore del labbro/naso ($\text{AU9}$).

2. **Arousal Psicofisiologico ($A \in [0.0, 1.0]$)**:
   $$A = 0.5 \cdot E_{\text{muscolare}} + 0.35 \cdot \text{Apertura Oculare Norm} + 0.25 \cdot \text{Dinamica Blink}$$
   - Misura l'attivazione energetica globale del sistema nervoso simpatico attraverso tono muscolare e apertura palpebrale.

3. **Indice di Carico Cognitivo ($CW \in [0.0, 1.0]$)** (Dinges & Metaxas 2005; Schleicher 2008):
   $$CW = (0.55 \cdot \text{AU4}_{\text{norm}}) + (0.25 \cdot \min(1.0, \Delta t_{\text{sustained}} / 5.0)) + \text{InibizioneBlink}$$
   - Cattura la co-attivazione di corrugamento persistente e inibizione fisiologica del tasso di ammiccamento ($< 10$ BPM durante intensa concentrazione visuo-spaziale).

---

## 4. Stati Psicologici Operativi di Volo

Sulla base della convergenza tra metriche oculari (PERCLOS, microsleeps) e coordinate dimensionali (Valence, Arousal, Carico Cognitivo), AURA-Face inferisce 7 stati operativi contestualizzati:

1. **`Calm Alert`**: Condizione nominale di riposo vigile. Valenza bilanciata, arousal medio ($0.30 - 0.50$), PERCLOS $< 8\%$.
2. **`Focused Flow`**: Alta concentrazione e basso sforzo percepito (docking, procedure delicate). Carico cognitivo moderato ($0.35 - 0.55$), inibizione del blink, valenza neutra/positiva, PERCLOS nullo.
3. **`Cognitive Strain`**: Sovraccarico cognitivo acuto o frustrazione operativa (Dinges & Metaxas 2005). AU4 elevato e sostenuto per $>5$ s, valenza negativa ($<-0.15$). Rischio omissione checklist e tunnel attention.
4. **`Genuine Engagement`**: Benessere e coesione dell'equipaggio. Presenza del sorriso autentico di Duchenne (co-attivazione simmetrica di AU12 zigomatico + AU6 orbicolare perioculare).
5. **`Drowsiness Impairment`**: Compromissione severa della vigilanza da sonnolenza circadiana (Wierwille 1994; Dinges 1998). PERCLOS $\ge 8\%-10\%$, durata blink $>350$ ms, ammiccamenti lenti ("droops"), microsleeps.
6. **`Hypovigilance`**: Calo di vigilanza e noia da compiti monotoni a bassa stimolazione sensoriale. Arousal molto basso ($<0.18$), sguardo fisso non variato, assenza di mimica.
7. **`Isolation Exhaustion`**: Quadro di esaurimento psicofisico e ritiro affettivo tipico delle missioni di lunga durata (Mars-500). Pattern pre-rest attivo (AU4 prolungato per oltre 3 minuti senza sorriso) combinato a calo costante della valenza.

