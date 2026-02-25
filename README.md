# Riconoscimento Stato Semaforico con OpenCV

[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-green?logo=opencv)](https://opencv.org/)
[![C++](https://img.shields.io/badge/C++-17-blue?logo=c%2B%2B)](https://isocpp.org/)

## Panoramica

Questo progetto implementa una pipeline completa di image processing per il riconoscimento affidabile dello stato dei semafori (🔴 **Rosso**, 🟡 **Giallo**, 🟢 **Verde**). 
E' stato validato su dataset reali raccolti nel centro cittadino di **Cassino (FR)**, dimostrando efficacia sia di giorno che di notte.

**Obiettivi principali:**
- Rilevamento accurato delle luci semaforiche in diverse condizioni ambientali
- Classificazione robusta dello stato per supportare decisioni di guida
- Gestione di sfide come illuminazione variabile

---

## Caratteristiche Principali

### Pre-elaborazione Avanzata
- **Conversione colore RGB → HSV** per una segmentazione più efficace
- **Correzione gamma** per il miglioramento in condizioni di scarsa illuminazione
- **Bilanciamento della luminosità** tramite regolazione pixel-pixel

### Segmentazione Intelligente
- **Binarizzazione adattiva**:
  - Metodo *Triangle* per scenari diurni
  - Metodo *Otsu* per scenari notturni
- **Pulizia morfologica** con operazioni di erosione e apertura
- **Rilevamento contorni** tramite `cv::findContours`

### Filtraggio Multi-Stadio
- **Criteri geometrici**: area, perimetro, rapporto d'aspetto
- **Analisi cromatica**: verifica tonalità e saturazione HSV
- **Proprietà morfologiche**: calcolo della circolarità per forme rotondeggianti

### Classificazione Robusta
- **Identificazione colore** tramite intervalli HSV calibrati
- **Sistema di votazione ponderata** per decisione finale affidabile
- **Gestione casi ambigui** con soglie multiple di confidenza
Un sistema avanzato di **computer vision** per il rilevamento e la classificazione in tempo reale dello stato dei semafori urbani.


---

## Tecnologie Utilizzate

- **Linguaggio**: C++ 17
- **Libreria Computer Vision**: OpenCV 4.8.0
- **Ambiente di Sviluppo**: Visual Studio 2022
- **Elaborazione Immagini**: Tecniche di segmentazione, filtraggio morfologico, analisi dei contorni

---

## Esempi di riconoscimento:

<img width="403" height="220" alt="Screenshot 2025-10-08 123407" src="https://github.com/user-attachments/assets/5368b084-53dc-44c7-bbec-17567f90471e" />
<img width="392" height="221" alt="Screenshot 2025-10-08 123348" src="https://github.com/user-attachments/assets/95213cde-6089-4b93-a8cc-fd6e28b2c396" />
<img width="381" height="205" alt="image" src="https://github.com/user-attachments/assets/236f059e-8b85-4179-b95a-5abcdd067569" />
<img width="385" height="213" alt="Screenshot 2025-10-08 123425" src="https://github.com/user-attachments/assets/d3c8ec97-db69-4bfe-8a1a-919158cf032b" />

## Installazione ed Esecuzione

### Prerequisiti
- **OpenCV 4.8.0** o superiore
- **Compilatore C++17** 
- **CMake 3.12** o superiore
- **Sistema operativo**: Windows, Linux o macOS

### Compilazione

```bash
# Clonare repository
git clone https://github.com/noemilatorre/Traffic_light_Recognition.git
cd Progetto giorno o cd Progetto notte
```

## Autori

**Noemi La Torre**

**Colacicco Nunziamaria**

- Email: latorre.noemi17@gmail.com 

---
*Questo progetto è stato sviluppato come parte del corso di Image Processing and Analysis presso l'Università degli Studi di Cassino e del Lazio Meridionale.*
