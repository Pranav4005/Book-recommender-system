import os
import sys
import pickle
import numpy as np
import streamlit as st

from book_recommendation_system.logger.log import logging
from book_recommendation_system.config.configuration import AppConfiguration
from book_recommendation_system.pipeline.training_pipeline import TrainingPipeline
from book_recommendation_system.exception.exception_handler import AppException


class Recommendation:
    def __init__(self, app_config=AppConfiguration()):
        try:
            self.recommendation_config = app_config.get_recommendation_config()

            # Load serialized files once
            self.model = pickle.load(
                open(self.recommendation_config.trained_model_path, "rb")
            )

            self.book_pivot = pickle.load(
                open(self.recommendation_config.book_pivot_serialized_objects, "rb")
            )

            self.final_rating = pickle.load(
                open(self.recommendation_config.final_rating_serialized_objects, "rb")
            )

            logging.info("All serialized objects loaded successfully!")

        except Exception as e:
            raise AppException(e, sys) from e

    # ---------------------------------------------------------
    # Fetch Book Poster URLs
    # ---------------------------------------------------------
    def fetch_poster(self, suggested_book_ids):
        try:
            poster_urls = []

            for book_id in suggested_book_ids:
                book_title = self.book_pivot.index[book_id]

                temp_df = self.final_rating[
                    self.final_rating["title"] == book_title
                ]

                if not temp_df.empty:
                    image_url = temp_df.iloc[0]["image_url"]
                    poster_urls.append(image_url)
                else:
                    poster_urls.append(
                        "https://via.placeholder.com/150x220.png?text=No+Image"
                    )

            return poster_urls

        except Exception as e:
            raise AppException(e, sys) from e

    # ---------------------------------------------------------
    # Recommend Books
    # ---------------------------------------------------------
    def recommend_book(self, book_name):
        try:
            books_list = []

            # Get selected book index
            book_index = np.where(
                self.book_pivot.index == book_name
            )[0][0]

            # Get nearest neighbors
            distances, suggestions = self.model.kneighbors(
                self.book_pivot.iloc[book_index, :].values.reshape(1, -1),
                n_neighbors=6
            )

            suggested_book_ids = suggestions[0]

            # Fetch posters
            poster_urls = self.fetch_poster(suggested_book_ids)

            # Fetch book names
            for book_id in suggested_book_ids:
                books_list.append(self.book_pivot.index[book_id])

            return books_list, poster_urls

        except Exception as e:
            raise AppException(e, sys) from e

    # ---------------------------------------------------------
    # Train Recommendation Engine
    # ---------------------------------------------------------
    def train_engine(self):
        try:
            obj = TrainingPipeline()
            obj.start_training_pipeline()

            st.success("Training Completed Successfully!")
            logging.info("Training completed successfully!")

        except Exception as e:
            raise AppException(e, sys) from e

    # ---------------------------------------------------------
    # Display Recommendations
    # ---------------------------------------------------------
    def recommendations_engine(self, selected_book):
        try:
            recommended_books, poster_urls = self.recommend_book(
                selected_book
            )

            st.subheader("Recommended Books")

            cols = st.columns(5)

            for i in range(1, 6):

                with cols[i - 1]:
                    st.text(recommended_books[i])

                    st.image(
                        poster_urls[i],
                        width=150
                    )

        except Exception as e:
            raise AppException(e, sys) from e


# =============================================================
# Streamlit App
# =============================================================
if __name__ == "__main__":

    st.set_page_config(
        page_title="Book Recommender System",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 End-to-End Book Recommender System")

    st.write(
        "This is a Collaborative Filtering based Book Recommendation System."
    )

    obj = Recommendation()

    # ---------------------------------------------------------
    # Train Model Button
    # ---------------------------------------------------------
    if st.button("Train Recommender System"):
        obj.train_engine()

    # ---------------------------------------------------------
    # Load Book Names
    # ---------------------------------------------------------
    book_names = pickle.load(
        open(
            os.path.join("templates", "book_names.pkl"),
            "rb"
        )
    )

    # ---------------------------------------------------------
    # Select Book
    # ---------------------------------------------------------
    selected_book = st.selectbox(
        "Type or Select a Book",
        book_names
    )

    # ---------------------------------------------------------
    # Recommendation Button
    # ---------------------------------------------------------
    if st.button("Show Recommendation"):
        obj.recommendations_engine(selected_book)