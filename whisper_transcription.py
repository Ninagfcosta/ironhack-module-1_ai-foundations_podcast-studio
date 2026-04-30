import os
from openai import OpenAI
from pydub import AudioSegment
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def transcrever_simples(caminho_audio):
    print("Transcrevendo o áudio (simples)...")
    with open(caminho_audio, "rb") as arquivo:
        resultado = client.audio.transcriptions.create(
            model="whisper-1",
            file=arquivo,
            response_format="verbose_json"
        )
    salvar_json(resultado, "transcricoes/simples.json")
    salvar_txt(resultado.text, "transcricoes/simples.txt")
    print("Pronto! Resultado salvo em transcricoes/simples.txt")
    return resultado

def transcrever_com_dicas(caminho_audio):
    print("Transcrevendo com dicas...")
    dicas = "Esta é uma reunião de negócios. Os participantes discutem inteligência artificial, machine learning, APIs e projetos de tecnologia."
    with open(caminho_audio, "rb") as arquivo:
        resultado = client.audio.transcriptions.create(
            model="whisper-1",
            file=arquivo,
            response_format="verbose_json",
            prompt=dicas
        )
    salvar_json(resultado, "transcricoes/com_dicas.json")
    salvar_txt(resultado.text, "transcricoes/com_dicas.txt")
    print("Pronto! Resultado salvo em transcricoes/com_dicas.txt")
    return resultado

def transcrever_sem_dicas(caminho_audio):
    print("Transcrevendo sem dicas...")
    with open(caminho_audio, "rb") as arquivo:
        resultado = client.audio.transcriptions.create(
            model="whisper-1",
            file=arquivo,
            response_format="verbose_json"
        )
    salvar_json(resultado, "transcricoes/sem_dicas.json")
    salvar_txt(resultado.text, "transcricoes/sem_dicas.txt")
    print("Pronto! Resultado salvo em transcricoes/sem_dicas.txt")
    return resultado

def transcrever_longo(caminho_audio, minutos_por_pedaco=10):
    print(f"Cortando o áudio em pedaços de {minutos_por_pedaco} minutos...")
    audio = AudioSegment.from_file(caminho_audio)
    ms_por_pedaco = minutos_por_pedaco * 60 * 1000
    pedacos = [audio[i:i+ms_por_pedaco] for i in range(0, len(audio), ms_por_pedaco)]
    todos_segmentos = []
    texto_completo = ""
    for numero, pedaco in enumerate(pedacos):
        offset_segundos = numero * minutos_por_pedaco * 60
        arquivo_temp = f"audio/pedaco_{numero}.mp3"
        pedaco.export(arquivo_temp, format="mp3")
        print(f"Transcrevendo pedaço {numero + 1} de {len(pedacos)}...")
        with open(arquivo_temp, "rb") as f:
            resultado = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json"
            )
        for segmento in resultado.segments:
            inicio = segmento.start + offset_segundos
            fim = segmento.end + offset_segundos
            texto = segmento.text
            todos_segmentos.append({"inicio": inicio, "fim": fim, "texto": texto})
            texto_completo += f"[{formatar_tempo(inicio)}] {texto}\n"
        os.remove(arquivo_temp)
    salvar_txt(texto_completo, "transcricoes/completo_com_horarios.txt")
    with open("transcricoes/completo_com_horarios.json", "w") as f:
        json.dump(todos_segmentos, f, ensure_ascii=False, indent=2)
    exportar_srt(todos_segmentos, "transcricoes/legendas.srt")
    print("Pronto! Arquivos salvos em transcricoes/")
    return todos_segmentos

def exportar_srt(segmentos, caminho):
    with open(caminho, "w") as f:
        for i, seg in enumerate(segmentos, 1):
            f.write(f"{i}\n")
            f.write(f"{formatar_tempo_srt(seg['inicio'])} --> {formatar_tempo_srt(seg['fim'])}\n")
            f.write(f"{seg['texto'].strip()}\n\n")
    print(f"Legenda SRT salva em {caminho}")

def formatar_tempo(segundos):
    h = int(segundos // 3600)
    m = int((segundos % 3600) // 60)
    s = int(segundos % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def formatar_tempo_srt(segundos):
    h = int(segundos // 3600)
    m = int((segundos % 3600) // 60)
    s = int(segundos % 60)
    ms = int((segundos % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def salvar_json(resultado, caminho):
    with open(caminho, "w") as f:
        json.dump(resultado.model_dump(), f, ensure_ascii=False, indent=2)

def salvar_txt(texto, caminho):
    with open(caminho, "w") as f:
        f.write(texto)

if __name__ == "__main__":
    AUDIO = "audio/reuniao.mp3"
    if not os.path.exists(AUDIO):
        print(f"ERRO: Coloca o arquivo de áudio em: {AUDIO}")
    else:
        print("=== INICIANDO TRANSCRIÇÕES ===\n")
        transcrever_simples(AUDIO)
        transcrever_com_dicas(AUDIO)
        transcrever_sem_dicas(AUDIO)
        transcrever_longo(AUDIO)
        print("\n=== TUDO PRONTO! Veja a pasta transcricoes/ ===")
