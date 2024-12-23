# Riconoscimento Stato Semaforico con OpenCV

## Descrizione
Questo progetto implementa un sistema avanzato per la rilevazione e il riconoscimento dello stato dei semafori urbani, utilizzando la libreria OpenCV in C++. Il sistema è stato testato su dataset raccolti nel centro della città di Cassino (FR) e include funzionalità per il riconoscimento delle luci semaforiche anche in condizioni di illuminazione ridotta, come durante la notte.

## Obiettivi
L'obiettivo principale è fornire un sistema che, tramite l'analisi delle immagini, riconosca accuratamente lo stato delle luci semaforiche (verde, giallo, rosso). Questo è essenziale per supportare veicoli a guida autonoma nel prendere decisioni adeguate durante la marcia:
- **Verde:** Proseguire.
- **Giallo:** Rallentare.
- **Rosso:** Fermarsi.

## Caratteristiche del Progetto
- **Pre-elaborazione delle immagini:**
  - Conversione dallo spazio colore RGB a HSV per una segmentazione più efficace.
  - Correzione gamma per migliorare la visibilità in condizioni di scarsa illuminazione.
  - Incremento della luminosità tramite tecniche di regolazione pixel.

- **Segmentazione e rilevamento:**
  - Segmentazione basata sulla binarizzazione automatica con metodo Triangle nei video di giorni e metodo Otsu per quelli di notte.
  - Filtraggio di rumore e artefatti con operazioni morfologiche (Erosione, Apertura).
  - Identificazione dei contorni tramite `cv::findContours`.

- **Filtraggio avanzato:**
  - Criteri geometrici (area, perimetro, rapporto d'aspetto).
  - Verifica della tonalità e della luminosità.
  - Calcolo della circolarità per riconoscere forme arrotondate (luci semaforiche).

- **Classificazione dello stato del semaforo:**
  - Riconoscimento dei colori (verde, giallo, rosso) tramite intervalli HSV.
  - Votazione ponderata per stabilire lo stato corrente in modo robusto.

## Tecnologie Utilizzate
- **Libreria:** OpenCV
- **Linguaggio:** C++
- **Ambiente di sviluppo:** Visual Studio Code 2022

## Dataset
Il progetto è stato testato su video raccolti nel centro di Cassino (FR), in particolare su sei incroci principali.

