from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import subprocess
import os
from pathlib import Path
import whisper
import glob
from openai import OpenAI
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


# hh:mm:ss -> second
def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

# 비디오를 입력받은 timeline에 따라 분할
def split_video_at_times(video_path, times, subtitles, file_name, output_path):
    clip = VideoFileClip(video_path)
    total_duration = int(clip.duration)
    
    times.append(str(total_duration // 3600).zfill(2) + ':' +
                 str((total_duration % 3600) // 60).zfill(2) + ':' +
                 str(total_duration % 60).zfill(2))
        
    Path(output_path).mkdir(parents=True, exist_ok=True)
    for i in range(len(times) - 1):
        start = time_to_seconds(times[i])
        end = time_to_seconds(times[i+1])
        target_filename = output_path + '/' + f"{file_name}_{times[i].replace(':','')}_{subtitles[i]}.mp4"
        temp_path = f"static/video/clip_{i+1}.mp4"
        ffmpeg_extract_subclip(video_path, start, end, targetname=temp_path)
        os.rename(temp_path, target_filename)
        print(target_filename)


# 지정된 경로 내의 모든 비디오에서 오디오 추출
def extract_audio_from_videos(directory_path, save_path, output_format='mp3'):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        output_file_path = os.path.join(save_path, os.path.splitext(filename)[0] + '.' + output_format)
        command = f'ffmpeg -i "{file_path}" -vn -ab 128k -ar 44100 -y "{output_file_path}"'
        subprocess.run(command, shell=True)
        print(f'Extracted audio to {output_file_path}')


# STT
def transcribe_audio_and_save_to_file(model, audio_path, output_file):
    result = model.transcribe(audio_path)
    transcribed_text = result["text"]
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcribed_text)
    print(f"Transcription saved to {output_file}")


# 지정된 경로 내의 모든 오디오 파일에 대해 STT
def transcribe_all_audio_in_directory(input_directory, output_directory):
    # 입력 디렉토리 내의 모든 파일 확인
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')):  # 지원되는 오디오 파일 확장자
            audio_path = os.path.join(input_directory, filename)
            output_file = os.path.join(output_directory, os.path.splitext(filename)[0] + '.txt')
            transcribe_audio_and_save_to_file(model, audio_path, output_file)

# 경로 생성
def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"경로가 생성되었습니다: {path}")
    else:
        print(f"경로가 이미 존재합니다: {path}")


# GPT 요약
def documentify_gpt4(text, system_prompt):
    response = client.chat.completions.create(
  model="gpt-4-turbo-preview",
  messages=[
    {
      "role": "system",
      "content": system_prompt
    },
    {
      "role": "user",
      "content": text
    },
  ],
  temperature=0,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
    text_result = response.choices[0].message.content
    return text_result

# 문서화 진행
def documentify_batch(source_path, destination_path, system_prompt):
    create_path_if_not_exists(destination_path)

    for file_path in glob.glob(os.path.join(source_path, '*.txt')):
        file_name = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            document = documentify_gpt4(content, system_prompt)
            print(document)
        with open(os.path.join(destination_path, file_name), 'w', encoding='utf-8') as file:
            file.write(document)

# 지정된 경로 내의 모든 txt파일에 대해 문서화 진행
def documentify_batch_total(source_path, destination_path, system_prompt):
    create_path_if_not_exists(destination_path)
    content_sum = ''
    for file_path in glob.glob(os.path.join(source_path, '*.txt')):
        file_name = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            content_sum = content_sum + '\n' + '\n' + content

    print(content_sum)
    document = documentify_gpt4(content_sum, system_prompt)
    print(document)
    with open(os.path.join(destination_path, file_name), 'w', encoding='utf-8') as file:
        file.write(document)

# 지정된 경로의 모든 txt에서 파일명과 내부 텍스트를 list로 반환
def get_filenames_texts_from_dir(path):
    txt_files = glob.glob(os.path.join(path, '*.txt'))

    filenames = []
    filetexts = []

    for filename in txt_files:
        filenames.append(os.path.basename(filename))
        with open(filename, 'r', encoding='utf-8') as file:
            filetexts.append(file.read())
    return filenames, filetexts

# 지정된 경로의 모든 txt에서 내부 텍스트를 list로 반환
def get_texts_from_dir(path):
    txt_files = glob.glob(os.path.join(path, '*.txt'))
    filetexts = []
    for filename in txt_files:
        with open(filename, 'r', encoding='utf-8') as file:
            filetexts.append(file.read())
    return filetexts

# 파일명에서 timestamp 추출
def get_timestamp_topic(filename):
    filename_wo_extension = filename.rsplit('.', 1)[0]
    parts = filename_wo_extension.split('_')
    real_name = "".join(parts[:-2])
    timestamp = parts[-2]
    topic = parts[-1]
    return real_name, timestamp, topic

# 파일명 리스트로부터 파일명, timeline 추출
def get_timestamps_topics(filenames):
    timestamps = []
    topics = []
    realnames = []
    for i in filenames:
        realname, timestamp, topic = get_timestamp_topic(i)
        realnames.append(realname)
        timestamps.append(timestamp)
        topics.append(topic)
    return realnames, timestamps, topics

# hhmmss를 second로 변환
def convert_to_seconds(time_str):
    # 입력 문자열에서 시, 분, 초를 추출합니다.
    hours = int(time_str[0:2])
    minutes = int(time_str[2:4])
    seconds = int(time_str[4:6])

    # 초로 변환합니다: (시간 * 3600) + (분 * 60) + 초
    total_seconds = (hours * 3600) + (minutes * 60) + seconds

    return total_seconds

# openai api 설정
API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)  # 수정된 부분


# youtube api에서 timeline을 제공하지 않아 수기 입력
times = ["00:00:00",
"00:01:01",
"00:01:51",
"00:06:37",
"00:14:25",
"00:18:05",
"00:21:20",
"00:23:35",
"00:36:30",
"00:43:01",
"00:45:24",
"01:30:54",
"01:33:47",
"01:44:02",
"02:02:19",
"02:09:31",
"02:22:33",
"02:26:20"]

subtitles = ["Introduction",
"Bingo Board",
"The Internet",
"TCP-IP",
"Ports",
"DNS",
"DHCP",
"HTTP",
"Inspect",
"Status Codes",
"HTML",
"Harvard Pep Squad Prank",
"Regular Expressions",
"CSS",
"Bootstrap",
"JavaScript",
"Autocomplete",
"Geolocation"]

# 비디오 파일명
file_name = "CS50x 2024 - Lecture 8 - HTML CSS JavaScript"
url = 'https://youtu.be/qIh5JEoKe_c'
video_path = "static/enc_video/" + file_name + ".mp4"


splited_video_dir = "static/video/" + file_name
splited_audio_path = "static/audio/" + file_name
splited_text_dir = "static/text/" + file_name
section_desc_path = splited_text_dir + '/section_desc'
total_desc_path = splited_text_dir + '/total_desc'
section_summary_path = splited_text_dir + '/section_summary'
total_summary_path = splited_text_dir + '/total_summary'

# 1. 비디오 분할
split_video_at_times(video_path, times, subtitles, file_name, splited_video_dir)

# 2. 분할 dir에서 오디오 추출
extract_audio_from_videos(splited_video_dir, splited_audio_path, output_format='mp3')

# 3. 오디오 분할 dir에서 STT
model = whisper.load_model("base", device="cuda")                      # Whisper 모델 로드
transcribe_all_audio_in_directory(splited_audio_path, splited_text_dir)

# 4. 소단원 설명 생성(커리큘럼)
system_prompt_desc = "50단어 미만의 경어체의 한글로 요약하는 기계."
documentify_batch(splited_text_dir, section_desc_path, system_prompt_desc)

# 5. 동영상 설명 생성(커리큘럼)
documentify_batch_total(section_desc_path, total_desc_path, system_prompt_desc)

# 6. 소단원 요약 생성(퀴즈)
system_prompt_section_summary = "150단어 미만의 경어체의 한글로된 교육 자료로 요약하는 기계."
documentify_batch(splited_text_dir, section_summary_path, system_prompt_section_summary)

# 5. 동영상 요약 생성(퀴즈)
system_prompt_total_summary = '''1. 경어체의 한글 교육자료로 요약한다.
2. 교육 소개나 취업, 활용, 선행 지식에 대한 부분은 제외한다.
3. 이론 중심의 교육자료로 생성한다.'''
documentify_batch_total(section_summary_path, total_summary_path, system_prompt_total_summary)

# # 6. 데이터 추출
filenames, filetexts = get_filenames_texts_from_dir(splited_text_dir)   # 파일명, 내부 텍스트
realnames, timestamps, topics = get_timestamps_topics(filenames)        # 영상이름, hhmmss, 소제목

section_desc_list = get_texts_from_dir(section_desc_path)
total_desc_list = get_texts_from_dir(total_desc_path)
total_sum_list = get_texts_from_dir(total_summary_path)
section_sum_list = get_texts_from_dir(section_summary_path)
seconds = []

for i in timestamps:
    seconds.append(convert_to_seconds(i))

# # 7. Firebase videos collection insert

# Firebase 프로젝트 초기화
cred = credentials.Certificate('codeclimx-20240307-firebase-adminsdk-hkmxz-c03ddc16cb.json')  # 서비스 계정 키 파일 경로
firebase_admin.initialize_app(cred)

# Firestore 클라이언트 인스턴스 생성
db = firestore.client()

totalTranscript = ''
for i in filetexts:
    totalTranscript += i

videoName = realnames[0]


# Videos 컬렉션에 문서 추가
video_data = {
    'videoName': videoName,
    'url': url,
    'totalTranscript': totalTranscript,
    'totalDesc': total_desc_list[0],
    'quiz_generated' : False,
    'detail' : total_sum_list[0],
    'descriptive_generated' : False
}
video_ref = db.collection('videos').document()
video_ref.set(video_data)

sections_data = []
for filename, filetext, realname, timestamp, topic, second, section_desc, section_sum in zip(filenames, filetexts, realnames, timestamps, topics, seconds, section_desc_list, section_sum_list):
# Sections 하위 컬렉션에 문서 추가
  section = {
          'id': topic,
          'text': filetext,
          'timestamp': timestamp,
          'second' : second,
          'section_desc' : section_desc,
          'url' : url,
          'sectionSum' : section_sum
      }
  sections_data.append(section)

# 각 섹션 데이터를 Sections 하위 컬렉션에 추가
for section in sections_data:
    # 자동 생성된 문서 ID를 사용하여 섹션을 추가
    video_ref.collection('sections').add(section)