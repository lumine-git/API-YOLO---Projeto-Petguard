from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import math
import datetime
from collections import defaultdict, Counter

app = FastAPI(title="PetGuard AI Backend", version="1.0.0")

# 1. Libera o acesso para o aplicativo mobile do seu grupo (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite requisições de qualquer origem/dispositivo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Carrega o modelo YOLO ao iniciar o servidor
model = YOLO('yolov8m.pt')

# Variáveis para manter o histórico entre as chamadas do app
history_classes = defaultdict(list)
ultima_refeicao_dt = datetime.datetime.now()
primeira_refeicao_registrada = False

# Configuração de alerta: 6 horas de inatividade sem comer
LIMITE_JEJUM_HORAS = 6.0

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

@app.post("/analisar-frame")
async def analisar_frame(file: UploadFile = File(...)):
    global ultima_refeicao_dt, primeira_refeicao_registrada

    # Lê a imagem/frame enviada pelo aplicativo
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Imagem inválida recebida."}

    # Executa a inferência do YOLOv8
    results = model.track(frame, persist=True, classes=[15, 16, 45], conf=0.30, verbose=False)

    status_atual = "Fora da zona de alimentação"
    esta_comendo = False
    bowls_centers = []
    pets = []

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        for box, cls_id, track_id in zip(boxes, clss, track_ids):
            class_name = model.names[int(cls_id)]
            if class_name in ['dog', 'cat']:
                history_classes[track_id].append(class_name)
                especie = Counter(history_classes[track_id]).most_common(1)[0][0]
                pets.append({"id": int(track_id), "especie": especie, "center": get_center(box)})
            elif class_name == 'bowl':
                bowls_centers.append(get_center(box))

        # Checa proximidade entre o pet e o pote
        for pet in pets:
            for b_center in bowls_centers:
                dist = math.hypot(pet['center'][0] - b_center[0], pet['center'][1] - b_center[1])
                if dist < 180:
                    esta_comendo = True
                    status_atual = "Alimentando-se"
                    ultima_refeicao_dt = datetime.datetime.now()
                    primeira_refeicao_registrada = True
                    break

    # Cálculo do tempo de jejum
    agora = datetime.datetime.now()
    tempo_sem_comer_horas = (agora - ultima_refeicao_dt).total_seconds() / 3600.0

    alerta_jejum = False
    mensagem_alerta = "Comportamento normal."

    if primeira_refeicao_registrada and tempo_sem_comer_horas > LIMITE_JEJUM_HORAS:
        alerta_jejum = True
        mensagem_alerta = f"Atenção! O pet está há {round(tempo_sem_comer_horas, 1)}h sem se alimentar."

    # Resposta em JSON que o app vai consumir para montar o layout
    return {
        "pet_info": {
            "status_atual": status_atual,
            "especie_predominante": pets[0]["especie"] if pets else "Não identificado"
        },
        "alimentacao": {
            "esta_comendo": esta_comendo,
            "ultima_refeicao": ultima_refeicao_dt.strftime("%H:%M:%S"),
            "horas_sem_comer": round(tempo_sem_comer_horas, 2)
        },
        "alertas": {
            "alerta_jejum_ativo": alerta_jejum,
            "mensagem": mensagem_alerta
        }
    }