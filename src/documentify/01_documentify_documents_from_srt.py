from datetime import timedelta
import os
from dotenv import load_dotenv
from openai import OpenAI


API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY) 

# hh:mm:ss 형식의 timestamp를 timedelta 형싱으로 변환
def str_to_timedelta(time_str):
    hours, minutes, seconds = map(int, time_str.split(":"))
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)

# 파일 내부 문자열 반환
def read_srt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

# srt 파일에서 인덱스, timestamp, 자막 추출
def parse_subtitles(s):
    sections = s.strip().split('\n\n')
    indexes = []
    timelines = []
    subtitles = []
    
    for section in sections:
        parts = section.split('\n', 2)
        indexes.append(parts[0].strip())
        timelines.append(parts[1].strip())
        subtitles.append(parts[2].strip())
        
    return indexes, timelines, subtitles

# GPT 문서화
def documentify_gpt4(text):
    response = client.chat.completions.create(
  model="gpt-4-turbo-preview",
  messages=[
    {
      "role": "system",
      "content": '''1. A machine that organizes documents into paragraphs with the eqivelent length of 384 characters each, maintaining a timestamp for each paragraph.
2. Respond in English.
3. YOU MUST Ensure the text avoids vague references such as 'it', 'this', or 'etc.', and instead directly states the subject matter.
4. DO NOT cut out that start and ending of a paragraph with ...
5. PLEASE specify the subject directly and avoid using pronouns!!!
6. For storing strings in JSON, please use ' instead of " inside paragraphs.
7. Output format: [{"timestamp": "01:02:12", "paragraph": "part of paragraphs"},{"timestamp": "01:04:26", "paragraph": "part of paragraphs"}]'''



    },
    {
      "role": "user",
      "content": text
    },
  ],
  temperature=1,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
    textSum = response.choices[0].message.content
    return textSum


# 디렉토리 내부 모든 txt 파일에 대해 문서화 진행
def batch_summarization(source_directory, target_directory):
    # 대상 디렉토리가 없으면 생성
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    cnt = 1
    # 원본 디렉토리 내의 모든 파일을 순회
    for filename in os.listdir(source_directory):
        # 파일 경로 결합
        file_path = os.path.join(source_directory, filename)
        
        # 텍스트 파일인 경우에만 처리
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as source_file:
                content = source_file.read()
            
            result_content = documentify_gpt4(content)
            
            print(result_content)

            # 대상 파일 경로 생성
            target_file_path = os.path.join(target_directory, filename)
            
            # 대상 경로에 파일 쓰기
            with open(target_file_path, 'w', encoding='utf-8') as target_file:
                target_file.write(result_content)
        cnt += 1
    print(f"모든 파일이 {target_directory}로 복사되었습니다.")

# 경로 생성
def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"경로가 생성되었습니다: {path}")
    else:
        print(f"경로가 이미 존재합니다: {path}")


# 비디오명, 경로 지정
video_name = 'Machine Learning in 2024 – Beginners Course'
srt_english_path = 'static/srt/' + video_name + '_english.srt'

directory_path = "static/srt/" + video_name
full_srt = read_srt_file(srt_english_path)
indexes, timelines, subtitles = parse_subtitles(full_srt)


# video section을 youtube api에서 제공하지 않아 수기 입력 처리
unit_times = ["00:00:00",
"00:03:13",
"00:10:39",
"00:38:54",
"00:45:48",
"01:00:59",
"01:08:04",
"01:23:38",
"01:36:56",
"02:00:20",
"02:15:37",
"02:33:44",
"02:39:54",
"02:45:59",
"02:54:39",
"03:03:39",
"03:14:00",
"03:32:14",
"03:34:31",
"04:01:24",
"04:10:10",
"04:15:54"]

if unit_times[0] == '00:00:00':
    del unit_times[0]

unit_times.append('9999:99:99')



unit_times_timedelta = []

for i in unit_times:
    unit_times_timedelta.append(str_to_timedelta(i))

start_time = []
end_time = []

for i in timelines:
    start_time.append(i.split(',000 --> ')[0])
    end_time.append((i.split(',000 --> ')[1]).split(',')[0])

start_time_timedelta = []
end_time_timedelta = []

for i, j in zip(start_time, end_time):
    start_time_timedelta.append(str_to_timedelta(i))
    end_time_timedelta.append(str_to_timedelta(j))


srt_part = []
for idx_u, (u, ut) in enumerate(zip(unit_times, unit_times_timedelta)):
    part_text = ''
    for s, st, e, et, t in zip(start_time, start_time_timedelta, end_time, end_time_timedelta, subtitles):
        if idx_u > 0 and st < unit_times_timedelta[idx_u - 1]:
            continue
        if st >= ut:  # 자막의 시작 시간이 현재 단위 시간보다 뒤에 있으면 이전 부분을 더 이상 추가하지 않음
            break
        part_text += '\n' + s + '\n' + t + '\n\n'
    if part_text:  # part_text에 자막이 하나라도 추가되었다면 srt_part에 추가
        srt_part.append(part_text)

print(len(srt_part))


# 디렉터리가 없다면 생성
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
print(srt_part)


# 리스트의 각 문자열을 별도의 파일로 저장
for index, string in enumerate(srt_part):
    # 파일 이름 설정. 예: file_0.txt, file_1.txt, ...
    
    # srt_english_path = os.path.join(directory_path, f"file_{"{:02}".format(index+1)}.txt")
    srt_english_path = os.path.join(directory_path, f"file_{index:02}.txt")
    # 파일에 문자열 쓰기
    with open(srt_english_path, 'w') as file:
        file.write(string)

# 문서화 진행
target_directory = "static/srt/" + video_name + "/summarization_english"
create_path_if_not_exists(target_directory)
batch_summarization(directory_path, target_directory)
