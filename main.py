from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import math
import datetime
import sqlite3
from collections import defaultdict, Counter

app = FastAPI(
    title="PetGuard AI Backend",
    description="API de Inteligência Computacional para Monitoramento Pet com YOLOv8",
    version="2.0.0"
)

# 1. Configuração de CORS (Liberado para conexão com app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Inicialização do Banco de Dados SQLite Local
def init_db():
    conn = sqlite3.connect("petguard_local.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS refeicoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            especie TEXT,
            duracao_segundos INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alertas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            tipo_alerta TEXT,
            mensagem TEXT,
            lido INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

init_db()

# 3. Modelo YOLO e Controle de Estado
model = YOLO('yolov8m.pt')
history_classes = defaultdict(list)
ultima_refeicao_dt = datetime.datetime.now()
primeira_refeicao_registrada = False
LIMITE_JEJUM_HORAS = 6.0

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

# --- ROTAS DA API ---

@app.get("/")
def home():
    return {
        "status": "Online",
        "sistema": "PetGuard AI Core",
        "versao": "2.0.0",
        "docs": "/docs"
    }

@app.post("/analisar-frame")
async def analisar_frame(file: UploadFile = File(...)):
    global ultima_refeicao_dt, primeira_refeicao_registrada

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"erro": "Imagem inválida."}

    results = model.track(frame, persist=True, classes=[15, 16, 45], conf=0.30, verbose=False)

    status_atual = "Fora da zona de alimentação"
    esta_comendo = False
    bowls_centers = []
    pets = []
    bounding_boxes_detectadas = []

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        for box, cls_id, conf, track_id in zip(boxes, clss, confs, track_ids):
            class_name = model.names[int(cls_id)]
            
            bounding_boxes_detectadas.append({
                "id": int(track_id),
                "classe": class_name,
                "confianca": round(float(conf), 2),
                "box": [round(float(b), 1) for b in box]
            })

            if class_name in ['dog', 'cat']:
                history_classes[track_id].append(class_name)
                especie = Counter(history_classes[track_id]).most_common(1)[0][0]
                pets.append({
                    "id": int(track_id),
                    "especie": especie,
                    "center": get_center(box),
                    "confianca": round(float(conf), 2)
                })
            elif class_name == 'bowl':
                bowls_centers.append(get_center(box))

        # Checa proximidade com o pote
        for pet in pets:
            for b_center in bowls_centers:
                dist = math.hypot(pet['center'][0] - b_center[0], pet['center'][1] - b_center[1])
                if dist < 180:
                    esta_comendo = True
                    status_atual = "Alimentando-se"
                    agora = datetime.datetime.now()
                    
                    # Registra a refeição no SQLite se for um novo evento
                    if not primeira_refeicao_registrada or (agora - ultima_refeicao_dt).total_seconds() > 300:
                        conn = sqlite3.connect("petguard_local.db")
                        cursor = conn.cursor()
                        cursor.execute(
                            "INSERT INTO refeicoes (timestamp, especie, duracao_segundos) VALUES (?, ?, ?)",
                            (agora.strftime("%Y-%m-%d %H:%M:%S"), pet["especie"], 60)
                        )
                        conn.commit()
                        conn.close()

                    ultima_refeicao_dt = agora
                    primeira_refeicao_registrada = True
                    break

    agora = datetime.datetime.now()
    tempo_sem_comer_horas = (agora - ultima_refeicao_dt).total_seconds() / 3600.0 if primeira_refeicao_registrada else 0.0

    alerta_jejum = False
    mensagem_alerta = "Comportamento normal."

    if primeira_refeicao_registrada and tempo_sem_comer_horas > LIMITE_JEJUM_HORAS:
        alerta_jejum = True
        mensagem_alerta = f"Atenção! Pet sem se alimentar há {round(tempo_sem_comer_horas, 1)}h."
        
        # Registra alerta no banco
        conn = sqlite3.connect("petguard_local.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO alertas (timestamp, tipo_alerta, mensagem) VALUES (?, ?, ?)",
            (agora.strftime("%Y-%m-%d %H:%M:%S"), "JEJUM_PROLONGADO", mensagem_alerta)
        )
        conn.commit()
        conn.close()

    return {
        "timestamp": agora.strftime("%Y-%m-%d %H:%M:%S"),
        "pet_info": {
            "status_atual": status_atual,
            "especie": pets[0]["especie"] if pets else "Não identificado",
            "confianca_deteccao": pets[0]["confianca"] if pets else 0.0
        },
        "alimentacao": {
            "esta_comendo": esta_comendo,
            "ultima_refeicao": ultima_refeicao_dt.strftime("%H:%M:%S") if primeira_refeicao_registrada else "Nenhum registro hoje",
            "horas_sem_comer": round(tempo_sem_comer_horas, 2)
        },
        "alertas": {
            "alerta_jejum_ativo": alerta_jejum,
            "mensagem": mensagem_alerta
        },
        "deteccoes_brutas": bounding_boxes_detectadas
    }

@app.get("/dashboard")
def get_dashboard():
    conn = sqlite3.connect("petguard_local.db")
    cursor = conn.cursor()
    
    # Conta refeições do dia
    hoje = datetime.datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(*) FROM refeicoes WHERE timestamp LIKE ?", (f"{hoje}%",))
    total_refeicoes_hoje = cursor.fetchone()[0]
    
    conn.close()

    return {
        "resumo_diario": {
            "total_refeicoes_hoje": total_refeicoes_hoje,
            "status_sistema": "Monitorando 24/7",
            "ultima_refeicao": ultima_refeicao_dt.strftime("%H:%M:%S") if primeira_refeicao_registrada else "Sem registros"
        }
    }

@app.get("/historico")
def get_historico():
    conn = sqlite3.connect("petguard_local.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, especie, duracao_segundos FROM refeicoes ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()

    historico = []
    for r in rows:
        historico.append({
            "id": r[0],
            "horario": r[1],
            "especie": r[2],
            "duracao_est_segundos": r[3]
        })

    return {"historico_refeicoes": historico}

@app.get("/alertas-pendentes")
def get_alertas():
    conn = sqlite3.connect("petguard_local.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, tipo_alerta, mensagem FROM alertas ORDER BY id DESC LIMIT 5")
    rows = cursor.fetchall()
    conn.close()

    alertas = []
    for r in rows:
        alertas.append({
            "id": r[0],
            "horario": r[1],
            "tipo": r[2],
            "mensagem": r[3]
        })

    return {"alertas_criticos": alertas}
