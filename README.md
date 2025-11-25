# KI_Basketballspieler_Erkennung


Dieses Projekt analysiert ein Video Bild für Bild, erkennt Objekte mithilfe eines KI-Modells und zeichnet Bounding Boxes um die erkannten Bereiche. Die ausgewerteten Frames werden im NiceGUI-Interface dargestellt.

Der Parameter „CONF“ legt fest, ab welcher Erkennungswahrscheinlichkeit eine Bounding Box gezeichnet wird (Standard: 0.50).



----------------------------------------------------------------



1\. Projekt vorbereiten (nach dem Entpacken der ZIP-Datei)



----------------------------------------------------------------



1\. Öffnen Sie ein Terminal / eine Eingabeaufforderung im Projektordner.



2\. Erstellen Sie ein virtuelles Environment (venv):



&nbsp;  python -m venv venv



3\. Aktivieren Sie das venv:



&nbsp;  Für Windows:

&nbsp;  venv\\Scripts\\activate



4\. Installieren Sie alle benötigten Libraries:



&nbsp;  pip install -r requirements.txt



Im venv müssen folgende Libraries verfügbar sein:



\* cv2

\* torch

\* ultralytics (YOLO)

\* threading (Standardbibliothek)

\* queue (Standardbibliothek)



----------------------------------------------------------------



2\. Nutzung



----------------------------------------------------------------



Passen Sie folgende Variablen in der main.py an:



VIDEO\_PATH = 'A:\\downloads2\\ball.mp4'

MODEL\_PATH = 'A:\\downloads2\\best.pt'



best.pt = schneller, aber ungenauer

best2.pt = langsamer, aber genauer

Wählen Sie das Modell, das für Ihre Anwendung sinnvoller ist.



----------------------------------------------------------------



3\. Video vorbereiten



----------------------------------------------------------------



1\. Suchen Sie auf YouTube ein beliebiges Basketball-Match.

2\. Nutzen Sie einen YouTube-zu-MP4-Konverter Ihrer Wahl.

3\. Achten Sie darauf, keine unerwünschte Software herunterzuladen.

4\. Speichern Sie das MP4-Video im gewünschten Ordner.

5\. Passen Sie VIDEO\_PATH entsprechend an.



----------------------------------------------------------------



4\. Hinweise



----------------------------------------------------------------



Das Projekt wurde mit PyCharm entwickelt und getestet.

Jede andere Python-IDE funktioniert ebenfalls, solange das venv korrekt aktiviert ist.


