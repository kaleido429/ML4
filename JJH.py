import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, LeakyReLU, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # 시각화를 위한 라이브러리
import os # 파일 및 디렉토리 작업을 위한 모듈 + GPU 설정
import random
import shutil # 폴더 및 파일 작업을 위한 모듈
import time
from datetime import datetime

# GPU 설정 및 확인
def setup_gpu():
    """
    GPU 사용 설정 및 확인
    TensorFlow가 GPU를 사용할 수 있도록 메모리 증가를 허용
    """
    # GPU 장치 확인
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # GPU 메모리 증가 허용 (필요에 따라 메모리 할당)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU 사용 가능: {len(gpus)}개의 GPU 감지됨")
            print(f"GPU 장치: {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"GPU 설정 오류: {e}")
    else:
        print("GPU를 사용할 수 없습니다. CPU로 실행됩니다.")
    
    # 현재 사용 중인 장치 확인
    print(f"TensorFlow 버전: {tf.__version__}")
    print(f"사용 가능한 GPU: {tf.config.list_physical_devices('GPU')}")

# 하이퍼파라미터 설정
class Config:
    """
    하이퍼파라미터 및 설정 클래스
    실험을 위한 다양한 설정값들을 관리
    """
    # 데이터셋 경로 설정
    BASE_DATASET_PATH = './caltech-101'  # Caltech-101 데이터셋 경로
    PROCESSED_DATA_PATH = './caltech_101_processed'
    
    # 이미지 전처리 파라미터
    IMG_WIDTH = 256    # 이미지 너비 (계산 효율성을 위해 128로 설정)
    IMG_HEIGHT = 256   # 이미지 높이
    CHANNELS = 3       # RGB 채널
    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    
    # 훈련 하이퍼파라미터
    BATCH_SIZE = 32    # 미니배치 크기: GPU 메모리와 수렴 안정성 고려
    EPOCHS = 75        # 최대 에포크 수 (Early Stopping 사용)
    LEARNING_RATE = 0.001  # 학습률: Adam optimizer의 기본값
    
    # 데이터 분할 비율
    TRAIN_RATIO = 0.8  # 훈련 데이터 80%
    VAL_RATIO = 0.1    # 검증 데이터 10%
    TEST_RATIO = 0.1   # 테스트 데이터 10%
    
    # 실험 설정
    NUM_TRIALS = 1     # 1회만 실행으로 변경

def create_directory_structure():
    """
    데이터셋 디렉토리 구조 생성
    훈련/검증/테스트용 폴더를 생성하고 데이터를 분할
    """
    config = Config()
    
    # 기존 처리된 데이터 제거 (새로운 분할을 위해)
    if os.path.exists(config.PROCESSED_DATA_PATH):
        shutil.rmtree(config.PROCESSED_DATA_PATH)
        print(f"기존 처리된 데이터 제거: {config.PROCESSED_DATA_PATH}")
    
    # 새 디렉토리 생성
    train_dir = os.path.join(config.PROCESSED_DATA_PATH, 'train')
    val_dir = os.path.join(config.PROCESSED_DATA_PATH, 'validation')
    test_dir = os.path.join(config.PROCESSED_DATA_PATH, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 원본 데이터셋 확인
    if not os.path.exists(config.BASE_DATASET_PATH):
        raise FileNotFoundError(f"데이터셋을 찾을 수 없습니다: {config.BASE_DATASET_PATH}")
    
    # 카테고리별 데이터 분할
    categories = []
    for item in os.listdir(config.BASE_DATASET_PATH):
        item_path = os.path.join(config.BASE_DATASET_PATH, item)
        if os.path.isdir(item_path):
            # 이미지 파일이 있는 폴더만 포함
            images = [f for f in os.listdir(item_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(images) >= 3:  # 최소 3개 이상의 이미지가 있는 카테고리만 포함
                categories.append(item_path)
    
    print(f"총 {len(categories)}개의 카테고리 발견")
    
    # 각 카테고리별로 데이터 분할
    for category_path in categories:
        category_name = os.path.basename(category_path)
        
        # 각 분할에 대한 카테고리 폴더 생성
        os.makedirs(os.path.join(train_dir, category_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category_name), exist_ok=True)
        
        # 이미지 목록 가져오기 및 셔플
        images = [f for f in os.listdir(category_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        
        # 데이터 분할 인덱스 계산
        n_images = len(images)
        train_end = int(config.TRAIN_RATIO * n_images)
        val_end = int((config.TRAIN_RATIO + config.VAL_RATIO) * n_images)
        
        # 데이터 분할
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        # 파일 복사
        for img in train_images:
            shutil.copy2(os.path.join(category_path, img), 
                        os.path.join(train_dir, category_name, img))
        for img in val_images:
            shutil.copy2(os.path.join(category_path, img), 
                        os.path.join(val_dir, category_name, img))
        for img in test_images:
            shutil.copy2(os.path.join(category_path, img), 
                        os.path.join(test_dir, category_name, img))
    
    print("데이터셋 분할 완료")
    return len(categories)

def create_data_generators():
    """
    데이터 제너레이터 생성
    훈련용: 데이터 증강 적용
    검증/테스트용: 정규화만 적용
    """
    config = Config()
    
    # 훈련용 데이터 증강 설정
    # 회전, 이동, 전단, 확대/축소, 수평 뒤집기를 통한 데이터 증강
    train_datagen = ImageDataGenerator(
        rescale=1./255,           # 픽셀값 정규화 [0,1]
        rotation_range=20,        # 20도 범위 내 회전
        width_shift_range=0.2,    # 수평 이동
        height_shift_range=0.2,   # 수직 이동
        shear_range=0.2,          # 전단 변환
        zoom_range=0.2,           # 확대/축소
        horizontal_flip=True,     # 수평 뒤집기
        fill_mode='nearest'       # 빈 공간 채우기 방법
    )
    
    # 검증/테스트용: 정규화만 적용
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # 데이터 제너레이터 생성
    train_generator = train_datagen.flow_from_directory(
        os.path.join(config.PROCESSED_DATA_PATH, 'train'),
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(config.PROCESSED_DATA_PATH, 'validation'),
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(config.PROCESSED_DATA_PATH, 'test'),
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

def create_cnn_model(input_shape, num_classes):
    """
    CNN 모델 생성 함수
    - 입력 이미지 크기와 클래스 수를 기반으로 모델 구성
    - Batch Normalization, Dropout, MaxPooling2D 레이어 포함
    Args:
        input_shape: 입력 이미지 크기 (height, width, channels)
        num_classes: 분류할 클래스 수
    Returns:
        model: 컴파일된 CNN 모델
    1. Conv2D 레이어를 사용하여 특징 추출
    2. Batch Normalization으로 학습 안정성 향상
    3. MaxPooling2D로 공간적 차원 축소
    4. Dropout으로 과적합 방지
    5. Flatten 후 Dense 레이어로 분류
    6. 최종 출력 레이어는 softmax 활성화 함수 사용
    """
    model = Sequential([
        # Convolutional Block 1 (큰 필터로 초기 특성 추출)
        Conv2D(64, (5, 5), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Convolutional Block 2
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Convolutional Block 3
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Convolutional Block 4
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Flatten -> Fully Connected
        Flatten(),
        
        # Dense Layer 1
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        # Dense Layer 2
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        # Dense Layer 3
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        # Output Layer
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def train_model_with_callbacks(model, train_generator, validation_generator, epochs):
    """
    콜백을 사용한 모델 훈련
    
    사용된 콜백:
    1. EarlyStopping: 검증 손실이 개선되지 않으면 조기 종료
    2. ReduceLROnPlateau: 검증 손실이 개선되지 않으면 학습률 감소
    
    Args:
        model: 훈련할 모델
        train_generator: 훈련 데이터 제너레이터
        validation_generator: 검증 데이터 제너레이터
        epochs: 최대 에포크 수
    
    Returns:
        훈련 히스토리
    """
    # Early Stopping 콜백
    # validation loss가 10 에포크 동안 개선되지 않으면 훈련 중단
    early_stopping = EarlyStopping(
        monitor='val_loss',      # 모니터링할 메트릭
        patience=10,             # 개선되지 않을 때 기다릴 에포크 수
        restore_best_weights=True,  # 최적 가중치 복원
        verbose=1
    )
    
    # Learning Rate Reduction 콜백
    # validation loss가 5 에포크 동안 개선되지 않으면 학습률을 0.2배로 감소
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',      # 모니터링할 메트릭
        factor=0.2,              # 학습률 감소 비율
        patience=5,              # 개선되지 않을 때 기다릴 에포크 수
        min_lr=1e-7,             # 최소 학습률
        verbose=1
    )
    
    # 모델 training with GPU acceleration
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
    
    return history

def evaluate_model_comprehensive(model, test_generator, class_labels):
    """
    포괄적인 모델 평가
    - 정확도, F1-score 계산
    - Confusion Matrix 생성
    - 클래스별 성능 분석
    
    Args:
        model: 평가할 모델
        test_generator: 테스트 데이터 제너레이터
        class_labels: 클래스 레이블 리스트
    
    Returns:
        accuracy, f1_score (macro average)
    """
    # 모델 예측
    print("모델 예측 중...")
    y_pred_probabilities = model.predict(test_generator, verbose=1)
    y_pred_classes = np.argmax(y_pred_probabilities, axis=1)
    
    # 실제 레이블
    y_true_classes = test_generator.classes
    
    # 메트릭 계산
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    f1_macro = f1_score(y_true_classes, y_pred_classes, average='macro')
    f1_weighted = f1_score(y_true_classes, y_pred_classes, average='weighted')
    
    print(f"테스트 정확도: {accuracy:.4f}")
    print(f"F1-score (macro): {f1_macro:.4f}")
    print(f"F1-score (weighted): {f1_weighted:.4f}")
    
    # Confusion Matrix 생성 및 저장
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # 클래스가 많을 경우 간단한 confusion matrix 플롯
    plt.figure(figsize=(12, 10))
    if len(class_labels) <= 20:  # 클래스가 20개 이하일 때만 레이블 표시
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    else:
        sns.heatmap(cm, cmap='Blues')
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curve 계산 및 플롯 (다중 클래스)
    print("ROC Curve 계산 및 플롯 중...")
    n_classes = len(class_labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # One-vs-Rest 전략
    y_true_binary = tf.keras.utils.to_categorical(y_true_classes, num_classes=n_classes)
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_pred_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Macro-average ROC Curve 계산
    # 모든 클래스의 FPR을 interpolate
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # 각 클래스의 TPR을 all_fpr에 interpolate
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # 평균 내기 및 AUC 계산
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # ROC Curve 플롯
    plt.figure(figsize=(8, 6))
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average ROC curve (area = {roc_auc["macro"]:.2f})',
             color='navy', linestyle=':', linewidth=4)
    
    # 각 클래스의 ROC Curve 플롯 (선택 사항)
    # for i in range(n_classes):
    #     plt.plot(fpr[i], tpr[i], lw=2,
    #              label=f'ROC curve of class {class_labels[i]} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return accuracy, f1_macro

def plot_training_history(history, trial_num=None):
    """
    훈련 히스토리 시각화
    - 정확도 곡선
    - 손실 곡선
    
    Args:
        history: 훈련 히스토리
        trial_num: 시도 번호 (선택사항)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 정확도 플롯
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 손실 플롯
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 파일명 설정
    filename = f'training_history_trial_{trial_num}.png' if trial_num else 'training_history.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"훈련 히스토리 플롯 저장: {filename}")

def main():
    """
    메인 실행 함수
    전체 실험 프로세스 관리
    """
    print("=== CNN 모델 훈련 시작 ===")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # GPU 설정
    setup_gpu()
    
    # 시드 설정 (재현 가능한 결과를 위해)
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 데이터셋 준비
    print("\n1. 데이터셋 준비 중...")
    num_classes = create_directory_structure()
    
    # 데이터 제너레이터 생성
    print("\n2. 데이터 제너레이터 생성 중...")
    train_generator, validation_generator, test_generator = create_data_generators()
    
    # 클래스 정보 확인
    class_labels = list(train_generator.class_indices.keys())
    print(f"총 클래스 수: {len(class_labels)}")
    print(f"훈련 샘플 수: {train_generator.samples}")
    print(f"검증 샘플 수: {validation_generator.samples}")
    print(f"테스트 샘플 수: {test_generator.samples}")
    
    # 하이퍼파라미터 실험 (선택사항)
    # hyperparameter_experiment()
    
    # 1회 실험
    print(f"\n3. 모델 훈련 시작...")
    
    print(f"\n--- 모델 훈련 및 평가 ---")
    
    # 모델 생성
    model = create_cnn_model(Config.INPUT_SHAPE, len(class_labels))
    
    # 모델 훈련
    start_time = time.time()
    history = train_model_with_callbacks(
        model, train_generator, validation_generator, Config.EPOCHS
    )
    training_time = time.time() - start_time
    
    print(f"훈련 시간: {training_time:.2f}초")
    
    # 모델 평가
    accuracy, f1 = evaluate_model_comprehensive(model, test_generator, class_labels)
    
    # 훈련 히스토리 저장
    plot_training_history(history, 1)
    
    # 최종 결과 출력
    print("\n=== 최종 결과 ===")
    print(f"테스트 정확도: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # 결과를 파일로 저장
    with open('experiment_results.txt', 'w', encoding='utf-8') as f:
        f.write("=== CNN 실험 결과 ===\n")
        f.write(f"데이터셋: Caltech-101\n")
        f.write(f"클래스 수: {len(class_labels)}\n")
        f.write(f"실험 횟수: 1\n")
        f.write(f"테스트 정확도: {accuracy:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"훈련 시간: {training_time:.2f}초\n")
    
    print(f"\n실험 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("결과 파일 저장: experiment_results.txt, confusion_matrix.png, training_history_trial_1.png")

if __name__ == '__main__':
    main()