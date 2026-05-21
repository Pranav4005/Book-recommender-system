from book_recommendation_system.pipeline.training_pipeline import TrainingPipeline


obj = TrainingPipeline()
obj.start_data_ingestion()
obj.start_data_validation()