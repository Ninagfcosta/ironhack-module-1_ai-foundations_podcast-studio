import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import jiwer

load_dotenv('/Users/Nina/lab-whisper/.env')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

AUDIO = '/Users/Nina/lab-whisper-evaluation/minha_voz.m4a'

def transcrever():
    print('Transcrevendo sua voz...')
    with open(AUDIO, 'rb') as f:
        resultado = client.audio.transcriptions.create(
            model='whisper-1',
            file=f,
            response_format='verbose_json'
        )
    with open('transcricoes/whisper.txt', 'w') as f:
        f.write(resultado.text)
    print(f'\nWhisper transcreveu:\n{resultado.text}')
    return resultado.text

def calcular_wer(ground_truth, whisper_text):
    print('\nCalculando WER...')
    transformacao = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    ref = transformacao(ground_truth)
    hip = transformacao(whisper_text)
    measures = jiwer.process_words(ref, hip)
    wer = measures.wer
    resultado = {
        'wer': round(wer, 4),
        'accuracy': round(1 - wer, 4),
        'wer_percentual': f'{wer*100:.2f}%',
        'accuracy_percentual': f'{(1-wer)*100:.2f}%',
        'substituicoes': measures.substitutions,
        'insercoes': measures.insertions,
        'delecoes': measures.deletions,
        'palavras_corretas': measures.hits,
    }
    print(f"WER: {resultado['wer_percentual']}")
    print(f"Precisao: {resultado['accuracy_percentual']}")
    with open('transcricoes/wer_results.json', 'w') as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    return resultado

def calcular_custo(duracao_segundos):
    preco_por_minuto = 0.006
    duracao_minutos = duracao_segundos / 60
    custo = duracao_minutos * preco_por_minuto
    cenarios = {
        '1_audio': round(custo, 6),
        '100_audios': round(custo * 100, 4),
        '1000_audios': round(custo * 1000, 4),
    }
    print(f"\nCusto por audio: ${cenarios['1_audio']}")
    print(f"100 audios/mes: ${cenarios['100_audios']}")
    print(f"1000 audios/mes: ${cenarios['1000_audios']}")
    with open('transcricoes/custo.json', 'w') as f:
        json.dump(cenarios, f, indent=2)
    return cenarios

def gerar_relatorio(whisper_text, ground_truth, wer, custo):
    relatorio = f"""# Relatorio de Avaliacao Whisper STT

## 1. Transcricao do Whisper
{whisper_text}

## 2. Ground Truth
{ground_truth}

## 3. Precisao
- WER: {wer['wer_percentual']}
- Precisao: {wer['accuracy_percentual']}
- Palavras corretas: {wer['palavras_corretas']}
- Substituicoes: {wer['substituicoes']}
- Insercoes: {wer['insercoes']}
- Delecoes: {wer['delecoes']}

## 4. Custo
- Por audio: ${custo['1_audio']}
- 100 audios: ${custo['100_audios']}
- 1000 audios: ${custo['1000_audios']}

## 5. Recomendacao
Precisao geral foi {wer['accuracy_percentual']}.
"""
    with open('transcricoes/relatorio.md', 'w') as f:
        f.write(relatorio)
    print('\nRelatorio salvo!')

if __name__ == '__main__':
    print('=== AVALIACAO DO WHISPER ===\n')
    whisper_text = transcrever()
    print('\nOuca seu audio e compare com o texto acima.')
    ground_truth = input('\nCole o texto corrigido (ou Enter se correto): ').strip()
    if not ground_truth:
        ground_truth = whisper_text
    with open('transcricoes/ground_truth.txt', 'w') as f:
        f.write(ground_truth)
    wer = calcular_wer(ground_truth, whisper_text)
    custo = calcular_custo(duracao_segundos=45)
    gerar_relatorio(whisper_text, ground_truth, wer, custo)
    print('\n=== TUDO PRONTO! ===')
