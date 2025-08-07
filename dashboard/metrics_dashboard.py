import json
import os
import time
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import tkinter as tk

# Используем бэкенд, который поддерживает GUI
matplotlib.use('TkAgg')


class MetricsHandler(FileSystemEventHandler):
    def __init__(self, file_path, refresh_interval=5):
        self.file_path = file_path
        self.refresh_interval = refresh_interval
        self.last_update = 0
        self.data = []
        self.window_closed = False  # Флаг закрытия окна

        # Настройка тёмной темы
        plt.style.use('default')
        self.bg_color = "#222222"
        self.fg_color = "#DDDDDD"

        # Создаем фигуру и оси
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.suptitle('Training Metrics Dashboard', fontsize=16, color=self.fg_color)
        self.fig.patch.set_facecolor(self.bg_color)
        for ax in self.axes:
            ax.set_facecolor(self.bg_color)
            ax.tick_params(colors=self.fg_color)
            ax.title.set_color(self.fg_color)
            ax.xaxis.label.set_color(self.fg_color)
            ax.yaxis.label.set_color(self.fg_color)

        # Увеличиваем отступ от левой границы
        self.fig.subplots_adjust(left=0.1)  # Увеличен левый отступ
        self.fig.subplots_adjust(hspace=0.4)

        # Обработчик закрытия окна
        self.fig.canvas.mpl_connect('close_event', self.handle_close)
        
        # Включаем интерактивный режим
        plt.ion()
        self.fig.show()
        self.fig.canvas.draw()

        # Убираем атрибут "topmost" для окна
        try:
            if matplotlib.get_backend().startswith('TkAgg'):
                win = self.fig.canvas.manager.window
                win.attributes('-topmost', False)
        except:
            pass

    def handle_close(self, event):
        """Обработчик события закрытия окна"""
        self.window_closed = True
        plt.close(self.fig)

    def on_modified(self, event):
        if event.src_path == self.file_path:
            current_time = time.time()
            if current_time - self.last_update > self.refresh_interval:
                self.last_update = current_time
                self.load_data()
                self.plot_metrics()

    def load_data(self):
        self.data = []
        try:
            if not os.path.exists(self.file_path):
                print(f"File {self.file_path} not found. Waiting...")
                return
                
            with open(self.file_path, 'r') as f:
                for line in f:
                    try:
                        self.data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"⚠️ Error loading data: {e}")

    def plot_metrics(self):
        # Не обновляем если окно закрыто
        if self.window_closed or not plt.fignum_exists(self.fig.number):
            return

        if not self.data:
            return

        try:
            df = pd.DataFrame(self.data)

            if 'step' not in df.columns:
                df['step'] = df.index

            numeric_cols = ['loss', 'eval_loss', 'learning_rate', 'grad_norm', 'epoch', 'step']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            train_df = df[df.get('step_type') == 'training']
            valid_df = df[df.get('step_type') == 'validation']

            # Loss
            ax = self.axes[0]
            ax.clear()
            ax.set_facecolor(self.bg_color)
            if not train_df.empty and 'loss' in train_df.columns:
                ax.plot(train_df['step'], train_df['loss'], 'b-', label='Training Loss', alpha=0.7)
            if not valid_df.empty and 'eval_loss' in valid_df.columns:
                ax.plot(valid_df['step'], valid_df['eval_loss'], 'r-', label='Validation Loss', alpha=0.7)
            ax.set_title('Loss', color=self.fg_color)
            ax.set_xlabel('Step', color=self.fg_color)
            ax.set_ylabel('Loss', color=self.fg_color)
            ax.legend()
            ax.grid(True, color='#444444')
            ax.tick_params(colors=self.fg_color)

            # Learning Rate
            ax = self.axes[1]
            ax.clear()
            ax.set_facecolor(self.bg_color)
            if not train_df.empty and 'learning_rate' in train_df.columns:
                ax.plot(train_df['step'], train_df['learning_rate'], 'g-', label='Learning Rate')
            ax.set_title('Learning Rate', color=self.fg_color)
            ax.set_xlabel('Step', color=self.fg_color)
            ax.set_ylabel('Learning Rate', color=self.fg_color)
            ax.legend()
            ax.grid(True, color='#444444')
            ax.tick_params(colors=self.fg_color)

            # Gradient Norm
            ax = self.axes[2]
            ax.clear()
            ax.set_facecolor(self.bg_color)
            if not train_df.empty and 'grad_norm' in train_df.columns:
                ax.plot(train_df['step'], train_df['grad_norm'], 'm-', label='Gradient Norm')
            ax.set_title('Gradient Norm', color=self.fg_color)
            ax.set_xlabel('Step', color=self.fg_color)
            ax.set_ylabel('Norm', color=self.fg_color)
            ax.legend()
            ax.grid(True, color='#444444')
            ax.tick_params(colors=self.fg_color)

            # Перерисовываем без поднятия окна
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        except Exception as e:
            print(f"⚠️ Error plotting metrics: {e}")


def main(metrics_file):
    event_handler = MetricsHandler(metrics_file)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(metrics_file), recursive=False)
    observer.start()

    print(f"📊 Monitoring metrics file: {metrics_file}")
    print("🔄 Dashboard will update automatically. Close window to exit.")

    try:
        event_handler.load_data()
        event_handler.plot_metrics()

        # Дадим окну время инициализироваться
        time.sleep(0.5)
        
        # Убираем фокус с окна после создания
        try:
            if matplotlib.get_backend().startswith('TkAgg'):
                win = event_handler.fig.canvas.manager.window
                win.attributes('-topmost', False)
                # Переводим фокус на предыдущее окно
                win.lower()
        except:
            pass

        # Поддерживаем окно открытым с меньшей частотой проверки
        while not event_handler.window_closed and plt.fignum_exists(event_handler.fig.number):
            plt.pause(1)  # Увеличиваем интервал проверки

    except KeyboardInterrupt:
        print("Stopping dashboard...")
    finally:
        print("Stopping observer...")
        observer.stop()
        observer.join()
        plt.ioff()
        plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Real-time Training Metrics Dashboard')
    parser.add_argument('--metrics-file', type=str, required=True,
                        help='Path to the metrics JSONL file')
    args = parser.parse_args()

    main(args.metrics_file)
