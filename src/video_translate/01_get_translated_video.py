import whisper
import os
from openai import OpenAI
from dotenv import load_dotenv
import os
import ast
import json
import time
import subprocess
import os
from pytube import YouTube
import uuid


# ffmpeg의 경로참조 실패 이슈에 대응하기 위해 중간 단계의 파일명을 임시파일로 두고 최종 원본파일명으로 수정 필요.
# 원본 동영상 경로와 중간단계의 temp 파일 경로 정리 필요


API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)  # 수정된 부분


def download_youtube_video(url, save_path):
    try:
        yt = YouTube(url)
        video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        video.download(save_path)
        print(f"동영상 다운로드 완료: {video.title}")
    except Exception as e:
        print("오류 발생:", e)

    return video.title.replace("'", "").replace(",", "")

def generate_file_id():
    file_id = uuid.uuid4()
    return str(file_id)

def rename_file(old_name, new_name):
    try:
        os.rename(old_name, new_name)
        print(f"파일명이 '{old_name}'에서 '{new_name}'으로 변경되었습니다.")
    except Exception as e:
        print(f"파일명 변경 중 오류 발생: {e}")





def extract_audio_from_video(video_file_path, audio_file_path):
    output_format='mp3'
    # ffmpeg를 사용하여 비디오로부터 오디오 추출
    command = f'ffmpeg -i "{video_file_path}" -vn -ab 128k -ar 44100 -y "{audio_file_path}"'
    subprocess.run(command, shell=True)

    print(f'Extracted audio to {audio_file_path}')



def seconds_to_srt_time_format(seconds):
    """초를 SRT 자막 형식의 시간 문자열로 변환합니다."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)

    # 시간, 분, 초를 두 자리 숫자로, 밀리초를 세 자리 숫자로 포맷팅
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"



def generate_subtitles(audio_path, output_file_path):
    """오디오 파일에서 자막을 생성합니다."""
    # Whisper 모델 로드
    model = whisper.load_model("base", device="cuda")
    
    # 오디오 파일에서 추론 실행
    result = model.transcribe(audio_path, verbose=False)
    
    # 결과에서 세그먼트 정보 추출
    segments = result["segments"]
    
    with open(output_file_path, "w", encoding='utf-8') as srt_file:
        for i, segment in enumerate(segments, start=1):
            start_time = seconds_to_srt_time_format(segment["start"])
            end_time = seconds_to_srt_time_format(segment["end"])
            text = segment["text"]
            
            srt_file.write(f"{i}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{text}\n\n")

    




def translate_english_to_korean_gpt4(text):
    response = client.chat.completions.create(
#   model="gpt-4-turbo-preview",
    model="gpt-4-turbo-preview",
  messages=[
    {
      "role": "system",
      "content": 
'''1. 입력받는 문자열은 10개의 key와 그 key에 속한 영문자막으로 구성된 json 양식의 문자열이다.
2. 영문 자막을 한글 경어체로 번역하되, IT전문용어는 번역하지 않는다.
3. 번역 과정에서 각 key에 속한 자막은 독립적으로 처리하며, 서로 다른 key에 속한 자막끼리는 결코 병합되거나 삭제되지 않는다.
4. 각 번역된 자막은 원본의 인덱스 번호(key)를 유지하며, 모든 key와 번역된 자막을 포함한 결과를 입력과 동일한 json 형식의 문자열로 출력한다.
5. 출력되는 key와 그 key에 속한 자막의 수는 입력되는 key의 수와 반드시 일치하는 10개 이어야 한다. 결과를 반환하기 전에 반드시 검증을 하고, 일치하지 않는 경우 다시 작성하여 반환한다.'''
    },
    
    {
      "role": "user",
      "content": text,
    },
  ],
  response_format={"type": "json_object"},
  temperature=1,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
    textTran = response.choices[0].message.content
    return textTran



def translate_english_to_korean_retry(response_before, text):
    response = client.chat.completions.create(
#   model="gpt-4-turbo-preview",
    model="gpt-4-turbo-preview",
  messages=[
    {
      "role": "system",
      "content": 
'''1. 입력받는 문자열은 10개의 key와 그 key에 속한 영문자막으로 구성된 json 양식의 문자열이다.
2. 영문 자막을 한글 경어체로 번역하되, IT전문용어는 번역하지 않는다.
3. 번역 과정에서 각 key에 속한 자막은 독립적으로 처리하며, 서로 다른 key에 속한 자막끼리는 결코 병합되거나 삭제되지 않는다.
4. 각 번역된 자막은 원본의 인덱스 번호(key)를 유지하며, 모든 key와 번역된 자막을 포함한 결과를 입력과 동일한 json 형식의 문자열로 출력한다.
5. 출력되는 key와 그 key에 속한 자막의 수는 입력되는 key의 수와 반드시 일치하는 10개 이어야 한다. 결과를 반환하기 전에 반드시 검증을 하고, 일치하지 않는 경우 다시 작성하여 반환한다.'''
    },
    {
      "role": "user",
      "content": text,
    },
    {
      "role": "assistant",
      "content": response_before
    },
    {
      "role": "user",
      "content": "너가 작성한 응답의 인덱스번호(key)의 갯수는 내가 요청한 jsondml 인덱스번호(key)의 갯수와 달라. 이것은 반드시 일치해야만 하고, 이점에 유의해서 다시 작성해줘. 최초의 요청과 동일한 json 형식으로만 응답해줘."
    },
  ],
  response_format={"type": "json_object"},
  temperature=1,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
    textTran = response.choices[0].message.content
    return textTran




def read_srt_file(file_path):
    # 파일을 열어 모든 내용을 읽은 후 반환
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content



def parse_subtitles(s):
    # Split the input string by double newlines to separate each subtitle section
    sections = s.strip().split('\n\n')
    
    # Initialize lists to store the indexes, timelines, and subtitles
    indexes = []
    timelines = []
    subtitles = []
    
    for section in sections:
        # Split each section by newline to separate index, timeline, and subtitle
        parts = section.split('\n', 2)
        
        # Append the parsed parts to their respective lists
        indexes.append(parts[0].strip())
        timelines.append(parts[1].strip())
        subtitles.append(parts[2].strip())
        
    return indexes, timelines, subtitles


def lists_to_json_string(keys, values):
    # 두 리스트의 요소를 결합하여 딕셔너리 생성
    combined_dict = dict(zip(keys, values))
    # 딕셔너리를 JSON 문자열로 변환
    json_string = json.dumps(combined_dict)
    return json_string

def json_string_to_lists(json_string):
    try:
        # JSON 문자열을 파이썬 딕셔너리로 변환
        parsed_json = json.loads(json_string)
    except json.JSONDecodeError:
        # JSON 형식이 잘못된 경우 오류 메시지를 출력하고 빈 리스트 반환
        print("입력된 문자열이 정확한 JSON 형식이 아닙니다.")
        return [], []
    keys = list(parsed_json.keys())
    values = list(parsed_json.values())
    
    return keys, values

def gradual_tranlate(indexes, subtitles):
    start_time = time.time()
    cnt = 1
    indexes_batch = []
    subtitles_batch = []
    indexes_tran = []
    subtitles_tran = []

    for i,j in zip(indexes, subtitles):
        indexes_batch.append(i)
        subtitles_batch.append(j)
        if cnt % 10 == 0 or cnt == len(indexes):
            json_before = lists_to_json_string(indexes_batch, subtitles_batch)
            json_after = translate_english_to_korean_gpt4(json_before)
            indexes_batch_tran, subtitles_batch_tran = json_string_to_lists(json_after)
            # 여기에 검증코드 필요
            cnt_tran = 1
            while len(indexes_batch_tran) != len(indexes_batch):
                print(f'번역 재시도 횟수 : {cnt_tran}')
                # json_after = translate_english_to_korean_gpt4(json_before)
                jason_retry = translate_english_to_korean_retry(json_after, json_before)
                indexes_batch_tran, subtitles_batch_tran = json_string_to_lists(jason_retry)
                print(jason_retry)
                # print(subtitles_batch_tran)
                cnt_tran += 1
            cnt_tran = 1
            print(indexes_batch_tran)
            print(subtitles_batch_tran)
            indexes_tran.extend(indexes_batch_tran)
            subtitles_tran.extend(subtitles_batch_tran)
            indexes_batch = []
            subtitles_batch = [] 
        cnt += 1
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"함수 실행 시간: {execution_time}초")

    return indexes_tran, subtitles_tran

def create_srt_file(indexes, timelines, subtitles, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for index, timeline, subtitle in zip(indexes, timelines, subtitles):
            file.write(f"{index}\n{timeline}\n{subtitle}\n\n")

def add_subtitle_to_video(video_file, subtitle_file, output_file):
    command = [
        'ffmpeg',
        '-i', video_file,  # 입력 비디오 파일
        '-vf', f"subtitles={subtitle_file}",  # 자막 파일
        '-c:v', 'libx264',  # 비디오 코덱 설정
        '-c:a', 'copy',  # 오디오 코덱 설정 (변경하지 않음)
        output_file  # 출력 파일
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"자막이 추가된 비디오가 성공적으로 생성되었습니다: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"비디오에 자막을 추가하는 데 실패했습니다: {e}")

def split_subtitles(srt_text):
    subtitles = srt_text.strip().split("\n\n")
    new_subtitles = []
    index = 1

    for subtitle in subtitles:
        parts = subtitle.split("\n", 2)  # 인덱스, 타임스팬, 텍스트로 분리
        timespan = parts[1]
        text = parts[2]

        start_time, end_time = timespan.split(" --> ")
        start_hours, start_minutes, start_seconds = [int(x) for x in start_time[:8].split(":")]
        start_milliseconds = int(start_time[9:])
        end_hours, end_minutes, end_seconds = [int(x) for x in end_time[:8].split(":")]
        end_milliseconds = int(end_time[9:])

        # 시간을 초 단위로 변환
        start_total_seconds = start_hours * 3600 + start_minutes * 60 + start_seconds + start_milliseconds / 1000
        end_total_seconds = end_hours * 3600 + end_minutes * 60 + end_seconds + end_milliseconds / 1000
        duration = end_total_seconds - start_total_seconds

        if len(text) > 60:
            parts_needed = len(text) // 60 + (1 if len(text) % 60 != 0 else 0)
            chars_per_part = len(text) // parts_needed
            time_per_part = duration / parts_needed

            for i in range(parts_needed):
                part_text = text[i * chars_per_part: (i + 1) * chars_per_part]
                part_start_time = start_total_seconds + i * time_per_part
                part_end_time = start_total_seconds + (i + 1) * time_per_part if i < parts_needed - 1 else end_total_seconds

                # 시작 및 종료 시간 형식 지정
                part_start_hours, part_start_remainder = divmod(part_start_time, 3600)
                part_start_minutes, part_start_seconds = divmod(part_start_remainder, 60)
                part_end_hours, part_end_remainder = divmod(part_end_time, 3600)
                part_end_minutes, part_end_seconds = divmod(part_end_remainder, 60)

                part_timespan = f"{int(part_start_hours):02d}:{int(part_start_minutes):02d}:{int(part_start_seconds):06.3f}".replace('.', ',') + " --> " + \
                                f"{int(part_end_hours):02d}:{int(part_end_minutes):02d}:{int(part_end_seconds):06.3f}".replace('.', ',')

                new_subtitles.append(f"{index}\n{part_timespan}\n{part_text}")
                index += 1
        else:
            new_subtitles.append(f"{index}\n{timespan}\n{text}")
            index += 1

    return "\n\n".join(new_subtitles)


def read_srt_file_to_string(file_path):
    # 파일을 열어 모든 내용을 읽은 후 반환
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def create_srt_file_from_string(filename, srt_text):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(srt_text)


# 1. 유튜브 링크로부터 동영상 다운로드
url_youtube = "https://youtu.be/jZzyERW7h1A?si=HTbotmJgZTMXWO3d"            
video_dir_path = "static/video"
audio_dir_path = "static/audio"
srt_dir_path = "static/srt"
video_name = download_youtube_video(url_youtube, video_dir_path)
print(video_name)
original_video_path = video_dir_path + '/' + video_name + '.mp4'
# 2. 원 파일명은 보존(video_name)하고 동영상 파일을 임시 파일명으로 변경
file_id = generate_file_id()
temp_video_path = video_dir_path + '/' + 'temp_' + file_id + '_before_encode' + '.mp4'
rename_file(original_video_path, temp_video_path)
print(original_video_path)
print(temp_video_path)


# 3. 임시 오디오, 자막_번역전, 자막_번역후, 동영상_인코딩후 파일명 생성
temp_audio_path = audio_dir_path + '/' + 'temp_' + file_id + '.mp3'
temp_srt_before_tran_path = srt_dir_path + '/' + 'temp_' + file_id + '_before_tran' + '.srt'
temp_srt_after_tran_path = srt_dir_path + '/' + 'temp_' + file_id + '.srt'
temp_video_after_encode_path = video_dir_path + '/' + 'temp_' + file_id + '_after_encode' + '.mp4'

# 4. 동영상파일로부터  mp3 파일 생성
extract_audio_from_video(temp_video_path, temp_audio_path)

# 5. mp3로부터 번역전_srt 생성
generate_subtitles(temp_audio_path, temp_srt_before_tran_path)

# 6. 번역전_srt로부터 번역후_srt파일 생성
srt_content = read_srt_file(temp_srt_before_tran_path)     
indexes, timelines, subtitles = parse_subtitles(srt_content)
indexes_tran, subtitles_tran = gradual_tranlate(indexes, subtitles)
create_srt_file(indexes, timelines, subtitles_tran, temp_srt_after_tran_path)

# 6.5. 번역후_srt파일에서 자막이 너무 길게 잡히는 구간은 자막과 구간을 분리 처리
srt_txt_path = temp_srt_after_tran_path
srt_txt_result_path = temp_srt_after_tran_path    
srt_text = read_srt_file_to_string(srt_txt_path)        
splited_srt = split_subtitles(srt_text)
create_srt_file_from_string(srt_txt_result_path, splited_srt)


# 7. 동영상 인코딩
add_subtitle_to_video(temp_video_path, temp_srt_after_tran_path, temp_video_after_encode_path)

# 8. 임시파일 제거 및 번역완료 srt 파일과 동영상파일 명칭을 변경
complete_video_path = video_dir_path + '/' + video_name + '.mp4'
complete_srt_path = srt_dir_path + '/' + video_name + '.srt'
complete_srt_path_en = srt_dir_path + '/' + video_name + '_english.srt'

rename_file(temp_video_after_encode_path, complete_video_path)
rename_file(temp_srt_after_tran_path, complete_srt_path)
rename_file(temp_srt_before_tran_path, complete_srt_path_en)

if os.path.exists(temp_video_path):
    os.remove(temp_video_path)
if os.path.exists(temp_audio_path):
    os.remove(temp_audio_path)
# if os.path.exists(temp_srt_before_tran_path):
#     os.remove(temp_srt_before_tran_path)
