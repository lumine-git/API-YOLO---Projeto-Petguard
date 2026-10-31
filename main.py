from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from ultralytics import YOLO
import cv2
import numpy as np
import math
import datetime
import sqlite3
from collections import defaultdict, Counter

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    global model
    model = YOLO('yolov8n.pt')
    yield

app = FastAPI(
    title="PetGuard AI Backend",
    description="API Completa com ROI, Controle de Sessão de Alimentação e Métricas de Atividade Física",
    version="4.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "petguard_local.db"
history_classes = defaultdict(list)

# Armazenamento de posições anteriores e sessões na memória do processo
posicoes_anteriores = {}  # {track_id: (x, y)}
sessoes_interacao = {}   # {track_id: {"tipo": "ALIMENTACAO", "inicio": dt, "ultima_vista": dt}}

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        cursor = conn.cursor()
        # Zonas de Interesse (ROI) configuradas no App pelo usuário
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS zonas_roi (
                tipo TEXT PRIMARY KEY, -- 'comida' ou 'agua'
                x1 REAL, y1 REAL, x2 REAL, y2 REAL
            )
        """)
        # Configuração legada de cores
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS configuracao_potes (
                tipo_pote TEXT PRIMARY KEY,
                cor_predominante TEXT
            )
        """)
        # Registro de Eventos (Alimentação, Hidratação, Atividade)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS registro_eventos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                pet_id INTEGER,
                especie TEXT,
                tipo_evento TEXT,
                duracao_segundos INTEGER,
                deslocamento_pixels REAL
            )
        """)
        # Alertas prioritários
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alertas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                categoria TEXT,
                nivel_prioridade TEXT,
                mensagem TEXT,
                lido INTEGER DEFAULT 0
            )
        """)
        conn.commit()

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def ponto_dentro_da_roi(ponto, roi):
    """Verifica se o centro do pet está dentro da caixa delimitadora da ROI [x1, y1, x2, y2]."""
    px, py = ponto
    return roi["x1"] <= px <= roi["x2"] and roi["y1"] <= py <= roi["y2"]

@app.post("/configurar-rois")
def configurar_rois(payload: dict = Body(...)):
    """
    Recebe as coordenadas normalizadas ou em pixels das zonas de comida e água.
    Exemplo payload:
    {
        "comida": {"x1": 100, "y1": 100, "x2": 250, "y2": 250},
        "agua": {"x1": 300, "y1": 100, "x2": 450, "y2": 250}
    }
    """
    with get_db() as conn:
        cursor = conn.cursor()
        for tipo, coords in payload.items():
            if tipo in ["comida", "agua"]:
                cursor.execute(
                    "INSERT OR REPLACE INTO zonas_roi (tipo, x1, y1, x2, y2) VALUES (?, ?, ?, ?, ?)",
                    (tipo, coords["x1"], coords["y1"], coords["x2"], coords["y2"])
                )
        conn.commit()
    return {"status": "Zonas de interesse (ROIs) configuradas com sucesso!"}

@app.post("/analisar-frame")
async def analisar_frame(file: UploadFile = File(...)):
    global posicoes_anteriores, sessoes_interacao

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Imagem inválida.")

    frame = cv2.resize(frame, (640, 640))
    results = model.track(frame, persist=True, classes=[15, 16, 45], conf=0.30, verbose=False)

    agora = datetime.datetime.now()
    pets = []
    
    # Busca ROIs salvas no banco
    rois = {}
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT tipo, x1, y1, x2, y2 FROM zonas_roi")
        for row in cursor.fetchall():
            rois[row["tipo"]] = {"x1": row["x1"], "y1": row["y1"], "x2": row["x2"], "y2": row["y2"]}

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        for box, cls_id, conf, track_id in zip(boxes, clss, confs, track_ids):
            class_name = model.names[int(cls_id)]

            if class_name in ['dog', 'cat']:
                history_classes[track_id].append(class_name)
                if len(history_classes[track_id]) > 30:
                    history_classes[track_id] = history_classes[track_id][-30:]
                especie = Counter(history_classes[track_id]).most_common(1)[0][0]

                centro_atual = get_center(box)
                
                # 1. CÁLCULO DE MOBILIDADE E ATIVIDADE FÍSICA
                deslocamento = 0.0
                if track_id in posicoes_anteriores:
                    prev_x, prev_y = posicoes_anteriores[track_id]
                    deslocamento = math.hypot(centro_atual[0] - prev_x, centro_atual[1] - prev_y)
                posicoes_anteriores[track_id] = centro_atual

                # Classifica o estado de atividade pontual
                if deslocamento < 2.0:
                    nivel_atividade = "Repouso / Inativo"
                elif deslocamento < 15.0:
                    nivel_atividade = "Atividade Moderada"
                else:
                    nivel_atividade = "Movimentação Intensa"

                # 2. DETECÇÃO DE INTERAÇÃO COM ZONAS DE ALIMENTAÇÃO (ROI)
                interagindo_tipo = None
                for tipo_zona, coords_roi in rois.items():
                    if ponto_dentro_da_roi(centro_atual, coords_roi):
                        interagindo_tipo = "ALIMENTACAO" if tipo_zona == "comida" else "HIDRATACAO"
                        break

                # 3. GERENCIAMENTO DE SESSÃO COM DEBOUNCE (Evita falsos registros picados)
                if interagindo_tipo:
                    if track_id not in sessoes_interacao:
                        sessoes_interacao[track_id] = {
                            "tipo": interagindo_tipo,
                            "inicio": agora,
                            "ultima_vista": agora
                        }
                    else:
                        sessoes_interacao[track_id]["ultima_vista"] = agora
                else:
                    # Se saiu da zona, verifica se a sessão expirou (mais de 60s longe)
                    if track_id in sessoes_interacao:
                        sessao = sessoes_interacao[track_id]
                        tempo_ausente = (agora - sessao["ultima_vista"]).total_seconds()
                        if tempo_ausente > 60:
                            duracao_total = int((sessao["ultima_vista"] - sessao["inicio"]).total_seconds())
                            if duracao_total >= 5:  # Considera evento válido apenas se durar mais de 5s
                                with get_db() as conn:
                                    cursor = conn.cursor()
                                    cursor.execute(
                                        "INSERT INTO registro_eventos (timestamp, pet_id, especie, tipo_evento, duracao_segundos, deslocamento_pixels) VALUES (?, ?, ?, ?, ?, ?)",
                                        (sessao["inicio"].strftime("%Y-%m-%d %H:%M:%S"), int(track_id), especie, sessao["tipo"], duracao_total, 0.0)
                                    )
                                    conn.commit()
                            del sessoes_interacao[track_id]

                pets.append({
                    "id": int(track_id),
                    "especie": especie,
                    "nivel_atividade": nivel_atividade,
                    "deslocamento_pixels": round(deslocamento, 2),
                    "interagindo": interagindo_tipo if interagindo_tipo else "Nenhum"
                })

    return {
        "timestamp": agora.strftime("%Y-%m-%d %H:%M:%S"),
        "pets_detectados": pets,
        "rois_ativas": list(rois.keys())
    }
