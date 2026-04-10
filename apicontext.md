================================================================================
CAISSIFY API - LOCAL DEVELOPMENT CONTEXT
================================================================================
Last Updated: March 11, 2026
Purpose: Quick reference for AI assistants and developers working on this project

================================================================================
PROJECT OVERVIEW
================================================================================

Caissify API is a Django REST Framework project for chess learning, tournaments,
puzzles, and player databases. It uses Docker with PostgreSQL, Redis, Celery
workers, and Celery beat for scheduled tasks.

Tech Stack:
- Django 5.2+ with Django REST Framework
- PostgreSQL 15 with Trigram extensions for fuzzy search
- Redis for caching and Celery message broker
- Celery for async tasks (workers + beat scheduler)
- Gunicorn as production WSGI server
- JWT authentication (SimpleJWT)
- Docker & Docker Compose for containerization

================================================================================
DOCKER SETUP
================================================================================

Docker Services (defined in docker-compose.yml):
1. postgres     - PostgreSQL 15 database (port 5433 externally)
2. redis        - Redis 7 for caching (port 6380 externally)
3. api          - Main Django API service (port 8000)
4. worker       - Celery worker for async tasks
5. beat         - Celery beat scheduler for periodic tasks
6. pgadmin      - Database admin UI (port 5050, optional profile: tools)
7. redis-commander - Redis UI (port 8081, optional profile: tools)

Container Names:
- caissify_postgres
- caissify_redis
- caissify_api
- caissify_worker
- caissify_beat

Starting the Application:
  docker-compose up -d                    # Start all services (detached)
  docker-compose up                       # Start with logs visible
  docker-compose up --build               # Rebuild and start
  docker-compose up --profile tools       # Include pgAdmin and redis-commander

Stopping the Application:
  docker-compose down                     # Stop all services
  docker-compose down -v                  # Stop and remove volumes (clears data)

View Logs:
  docker-compose logs -f api              # Follow API logs
  docker-compose logs -f worker           # Follow worker logs
  docker-compose logs -f beat             # Follow beat scheduler logs
  docker-compose logs --tail=100 api      # Last 100 lines

================================================================================
RUNNING COMMANDS INSIDE DOCKER CONTAINERS
================================================================================

General Pattern:
  docker-compose exec [service_name] [command]
  OR
  docker exec [container_name] [command]

Common Examples:

# Django Management Commands (inside API container):
  docker-compose exec api python manage.py migrate
  docker-compose exec api python manage.py makemigrations
  docker-compose exec api python manage.py createsuperuser
  docker-compose exec api python manage.py shell
  docker-compose exec api python manage.py test
  docker-compose exec api python manage.py collectstatic --noinput
  docker-compose exec api python manage.py runserver

# Interactive Shell:
  docker-compose exec -it api bash        # Bash shell in API container
  docker-compose exec -it api python manage.py shell  # Django shell
  docker-compose exec -it postgres psql -U caissify_user -d caissify_db

# Run Tests:
  docker-compose exec api python manage.py test                    # All tests
  docker-compose exec api python manage.py test books              # Specific app
  docker-compose exec api python manage.py test tests.fide.test_fide_search
  docker-compose exec api python tests/tournament/test_simple_tournament.py

# Custom Management Commands:
  docker-compose exec api python manage.py import_chess_book data/books/example.pgn

# Run Scripts:
  docker-compose exec api python scripts/testing/test_all_notifications.py

# Celery Commands (worker container):
  docker-compose exec worker celery -A core inspect active
  docker-compose exec worker celery -A core status

# Database Commands:
  docker-compose exec postgres pg_dump -U caissify_user caissify_db > backup.sql
  docker-compose exec -i postgres psql -U caissify_user caissify_db < backup.sql

# Working Directory Context (for i18n/translations):
  docker exec -w /app/databases caissify_api python ../manage.py makemessages -l es
  docker exec -w /app/fide caissify_api python ../manage.py compilemessages

================================================================================
PROJECT STRUCTURE
================================================================================

Root Files:
  manage.py              - Django management script
  docker-compose.yml     - Multi-service Docker configuration
  Dockerfile             - Python 3.12 multi-stage build
  docker-entrypoint.sh   - API container startup script (migrations + collectstatic + gunicorn)
  requirements.txt       - Python dependencies
  .env                   - Environment variables (not in git)
  .env.example           - Template for environment variables
  .env.docker            - Docker-specific environment variables

Core Django Apps:
  core/                  - Django project settings
    settings.py          - Main settings file
    urls.py              - Root URL configuration
    celery.py            - Celery configuration
    wsgi.py / asgi.py    - WSGI/ASGI application
    middleware/          - Custom middleware
    management/          - Custom management commands
    locale/              - Core translations

  users/                 - User authentication and profiles
    models.py            - Custom User model
    authentication.py    - Custom JWT authentication
    serializers.py       - User serializers
    views.py             - Auth endpoints
    tests.py             - User/auth tests

  fide/                  - FIDE player database
    models.py            - FIDEPlayer model
    filters.py           - Search filters
    cache_service.py     - Caching logic
    views.py             - FIDE player endpoints
    tests.py             - FIDE tests

  tournament/            - Tournament management
    models/              - Tournament models (split files)
    views/               - Tournament endpoints (split files)
    services/            - Business logic (pairing, standings)
    serializers/         - Response serializers
    tests.py             - Tournament tests

  lesson/                - Chess lessons and puzzles
    models/              - Lesson models
    views/               - Lesson endpoints
    services/            - Lesson business logic
    tests/               - Lesson tests

  databases/             - User chess game databases
    models/              - Split models:
      database.py        - ChessDatabase, DatabaseCollaborator
      game.py            - ChessGame
      share_token.py     - GameShareToken
      upload_job.py      - UploadJob
      position.py        - PositionStatistics
      position_index.py  - GamePositionIndex
      task_status.py     - TaskStatus
    views/               - Database endpoints
    services/            - Game processing logic
    tasks.py             - Celery tasks for async operations
    tests/               - Database tests
      test_database_limits.py
      test_position_filtering.py

  books/                 - Chess books and opening repertoires
    models/              - Split models:
      book.py, chapter.py, problem.py, progress.py, calibration.py
    views.py / views_v1.py - Book endpoints (v1 and v2)
    services/            - Book management logic
    tasks.py             - Celery tasks
    tests.py             - Book tests

  learn/                 - Chess learning module
  javafo/                - Additional chess features

Directories:
  tests/                 - Integration tests (organized by feature)
    fide/                - FIDE integration tests
    tournament/          - Tournament workflow tests
    lesson/              - Lesson tests

  scripts/               - Utility scripts
    analysis/            - Analysis scripts for business logic
    testing/             - Manual testing scripts

  docs/                  - Comprehensive documentation
    api/                 - API specifications
    fide/                - FIDE compliance docs
    tournament/          - Tournament system docs
    deployment/          - Deployment guides

  data/                  - Data files (books, FIDE data, etc.)
    books/               - Chess book PGN files
    fide/                - FIDE player data
    tournaments/         - Tournament data

  locale/                - Project-wide translations
  staticfiles/           - Collected static files (generated)
  mediafiles/            - User uploaded media
  cache/                 - Cache directory

================================================================================
MODEL ORGANIZATION
================================================================================

Pattern 1: Single models.py file (for simple apps)
  - Used in: fide/, users/
  - Example: fide/models.py contains FIDEPlayer model

Pattern 2: Split models/ directory (for complex apps)
  - Used in: databases/, books/, tournament/
  - Structure:
      app/models/
        __init__.py      - Import all models, define __all__
        model1.py        - Individual model file
        model2.py        - Individual model file
  
  Example (databases/models/):
    databases/models/__init__.py:
      from .database import ChessDatabase, DatabaseCollaborator
      from .game import ChessGame
      from .share_token import GameShareToken
      __all__ = ['ChessDatabase', 'DatabaseCollaborator', 'ChessGame', ...]
    
    databases/models/database.py:
      from django.db import models
      class ChessDatabase(models.Model):
          # Model definition

How to Edit Models:
1. Navigate to app/models.py or app/models/specific_model.py
2. Make changes to model fields/methods
3. Create migration: docker-compose exec api python manage.py makemigrations
4. Review migration: Check app/migrations/XXXX_migration_name.py
5. Apply migration: docker-compose exec api python manage.py migrate
6. Test: docker-compose exec api python manage.py test app_name

IMPORTANT: Always import models from the models package, not individual files:
  ✅ from databases.models import ChessDatabase
  ❌ from databases.models.database import ChessDatabase

================================================================================
MIGRATIONS
================================================================================

Creating Migrations:
  # Auto-detect changes to all apps
  docker-compose exec api python manage.py makemigrations

  # Specific app
  docker-compose exec api python manage.py makemigrations databases

  # With custom name
  docker-compose exec api python manage.py makemigrations databases --name add_upload_job

  # Empty migration (for data migrations or custom SQL)
  docker-compose exec api python manage.py makemigrations --empty databases

Running Migrations:
  # Apply all pending migrations
  docker-compose exec api python manage.py migrate

  # Specific app
  docker-compose exec api python manage.py migrate databases

  # Specific migration
  docker-compose exec api python manage.py migrate databases 0008

  # Show migration plan without running
  docker-compose exec api python manage.py migrate --plan

  # Show all migrations and status
  docker-compose exec api python manage.py showmigrations

Reverting Migrations:
  # Revert to previous migration
  docker-compose exec api python manage.py migrate databases 0007

  # Revert all migrations for an app
  docker-compose exec api python manage.py migrate databases zero

Migration Files Location:
  app/migrations/
    0001_initial.py
    0002_add_upload_job_model.py
    __init__.py

Migration Naming Convention:
  - Use descriptive names: add_field_name, remove_old_table, alter_user_email
  - Automatic names are acceptable for simple changes
  - Examples from project:
      databases/migrations/0002_add_upload_job_model.py
      databases/migrations/0006_add_external_source_tracking.py
      books/migrations/0002_problem_puzzle_last_move.py

Special Migration Cases:
  # PostgreSQL indexes (can't use CONCURRENTLY in migrations)
  # Create manually in psql or pgAdmin:
  CREATE INDEX CONCURRENTLY IF NOT EXISTS fideplayer_name_trgm_idx
  ON fide_fideplayer USING gin (name gin_trgm_ops);

  # Data migrations (use RunPython):
  from django.db import migrations
  
  def populate_data(apps, schema_editor):
      Model = apps.get_model('app_name', 'ModelName')
      # Perform data changes
  
  class Migration(migrations.Migration):
      operations = [
          migrations.RunPython(populate_data),
      ]

================================================================================
TESTING ORGANIZATION
================================================================================

Test File Locations:

1. Unit Tests (app-level):
   Location: app/tests.py or app/tests/
   Pattern: Django TestCase classes
   Run: docker-compose exec api python manage.py test app_name
   
   Examples:
     fide/tests.py
     fide/test_filters_comprehensive.py
     fide/test_top_players.py
     users/tests.py
     users/test_i18n.py
     users/test_board_themes.py
     books/tests.py
     tournament/tests.py
     lesson/tests.py
     lesson/tests/test_private_lessons.py
     lesson/tests/test_lichess_service.py
     databases/tests/test_database_limits.py
     databases/tests/test_position_filtering.py

2. Integration Tests (cross-app workflows):
   Location: tests/ (root level)
   Pattern: Organized by feature in subdirectories
   Run: docker-compose exec api python tests/feature/test_file.py
   
   Structure:
     tests/
       fide/
         test_fide_search.py
         test_fide_titles.py
         test_fide_compliant_pairing.py
         test_federation_endpoint.py
       tournament/
         test_simple_tournament.py
         test_tournament_workflow.py
         test_round_creation.py
         test_swiss_pairing.py
       integration/
         (cross-feature tests)

3. Other Test Files:
   - App root level: databases/test_encoding.py, databases/test_explorer.py
   - These are typically standalone test scripts or legacy tests

Test Patterns:

Unit Test Example (databases/tests/test_database_limits.py):
  from django.test import TestCase
  from rest_framework.test import APIClient
  from django.contrib.auth import get_user_model
  
  class DatabaseLimitsTestCase(TestCase):
      def setUp(self):
          self.client = APIClient()
          self.user = User.objects.create_user(...)
      
      def test_database_limit_enforced(self):
          self.client.force_authenticate(user=self.user)
          response = self.client.post('/api/v1/databases/', {...})
          self.assertEqual(response.status_code, 201)

Where to Put New Tests:

1. Testing a specific model/feature in one app?
   → Add to app/tests.py or app/tests/test_specific_feature.py

2. Testing API endpoints for one app?
   → Add to app/tests.py using APIClient

3. Testing workflow across multiple apps?
   → Add to tests/feature_name/test_workflow.py

4. Testing a new app?
   → Create app/tests.py or app/tests/ directory

Running Tests:

  # All tests
  docker-compose exec api python manage.py test

  # Specific app
  docker-compose exec api python manage.py test databases

  # Specific test file (Django style)
  docker-compose exec api python manage.py test databases.tests.test_database_limits

  # Specific test class
  docker-compose exec api python manage.py test databases.tests.test_database_limits.DatabaseLimitsTestCase

  # Specific test method
  docker-compose exec api python manage.py test databases.tests.test_database_limits.DatabaseLimitsTestCase.test_database_limit_enforced

  # Integration tests
  docker-compose exec api python tests/tournament/test_simple_tournament.py

  # With verbose output
  docker-compose exec api python manage.py test --verbosity=2

  # Keep test database
  docker-compose exec api python manage.py test --keepdb

================================================================================
ENVIRONMENT VARIABLES
================================================================================

Location: .env file (copy from .env.example)

Required Variables:
  DEBUG=True/False
  SECRET_KEY=your-secret-key
  DJANGO_SETTINGS_MODULE=core.settings
  
  # Database
  DB_NAME=caissify_db
  DB_USER=caissify_user
  DB_PASSWORD=caissify_password
  DB_HOST=postgres (inside Docker) or localhost (local dev)
  DB_PORT=5432 (inside Docker) or 5433 (external access)
  
  # Redis
  REDIS_URL=redis://redis:6379/0 (inside Docker)
  CELERY_BROKER_URL=redis://redis:6379/0
  CELERY_RESULT_BACKEND=redis://redis:6379/0
  
  # Email (Mailgun)
  MAILGUN_API_KEY=your-api-key
  MAILGUN_DOMAIN=your-domain
  DEFAULT_FROM_EMAIL=noreply@yourdomain.com
  
  # CORS
  FRONTEND_URL=http://localhost:5173
  ALLOWED_HOSTS=localhost,127.0.0.1,api.caissify.com
  
  # External Services
  EXTERNAL_GAMES_API_URL=http://127.0.0.1:8001
  
  # Firebase (for push notifications)
  FIREBASE_CREDENTIALS_PATH=firebase-credentials.json
  # OR
  FIREBASE_CREDENTIALS_JSON={"type":"service_account",...}

Docker Environment:
  - docker-compose.yml passes environment variables to containers
  - API, worker, and beat containers share same environment variables
  - Set DOCKER_ENV=true for Docker-specific behavior

================================================================================
API ENDPOINTS
================================================================================

Base URL: http://localhost:8000

Documentation:
  - Swagger UI: http://localhost:8000/swagger/
  - ReDoc: http://localhost:8000/redoc/
  - OpenAPI JSON: http://localhost:8000/swagger.json

Authentication:
  - JWT Bearer tokens (in Authorization header)
  - Format: Authorization: Bearer <access_token>
  - Get token: POST /api/auth/jwt/create/ {email, password}

Main Endpoint Patterns:
  /api/auth/          - Djoser authentication (register, login, etc.)
  /api/v1/fide/       - FIDE player search
  /api/v1/tournament/ - Tournament management
  /api/v1/lesson/     - Lessons and puzzles
  /api/v1/databases/  - Chess game databases
  /api/v1/books/      - Chess books and repertoires
  /api/v2/books/      - Books V2 (newer version)

Pagination:
  - All list endpoints are paginated
  - Default page size: 25
  - Query params: ?page=2, ?page_size=50

Filtering:
  - Uses django-filter
  - Example: /api/v1/fide/?name=carlsen&country=NOR

================================================================================
INSTALLED APPS & KEY PACKAGES
================================================================================

Django Apps (in INSTALLED_APPS):
  - users (custom user model)
  - fide (FIDE player database)
  - tournament (tournament management)
  - lesson (lessons and puzzles)
  - databases (chess game databases)
  - learn (learning module)
  - books (chess books and repertoires)

Third-Party Packages:
  - djangorestframework (REST API framework)
  - djangorestframework-simplejwt (JWT authentication)
  - djoser (user management endpoints)
  - django-cors-headers (CORS support)
  - django-filter (filtering)
  - drf-yasg (Swagger/OpenAPI docs)
  - celery (async task queue)
  - redis (cache and broker)
  - python-chess (chess logic and PGN parsing)
  - glicko2 (rating system)
  - psycopg2-binary (PostgreSQL adapter)
  - gunicorn (WSGI server)
  - django-anymail (email backend)
  - reportlab (PDF generation)

Custom Settings:
  - AUTH_USER_MODEL = 'users.User'
  - DEFAULT_PAGINATION_CLASS = PageNumberPagination
  - PAGE_SIZE = 25
  - ACCESS_TOKEN_LIFETIME = 7 days
  - CELERY_BROKER_URL = Redis

================================================================================
CELERY (ASYNC TASKS)
================================================================================

Celery Workers:
  - Container: caissify_worker
  - Processes background tasks (uploads, imports, notifications)
  - Command: celery -A core worker --loglevel=info

Celery Beat:
  - Container: caissify_beat
  - Schedules periodic tasks
  - Command: celery -A core beat --loglevel=info

Task Files:
  - app/tasks.py (e.g., databases/tasks.py, books/tasks.py)
  - Core config: core/celery.py

Example Tasks:
  - databases.tasks.bulk_upload_games_async
  - databases.tasks.create_database_from_fide_async
  - books.tasks.import_chess_book_async

Monitoring Celery:
  docker-compose logs -f worker              # View worker logs
  docker-compose logs -f beat                # View beat logs
  docker-compose exec worker celery -A core inspect active    # Active tasks
  docker-compose exec worker celery -A core status            # Worker status
  docker-compose exec worker celery -A core inspect stats     # Task stats

Task Configuration (core/settings.py):
  - Task routes: CELERY_TASK_ROUTES
  - Rate limits: CELERY_TASK_ANNOTATIONS
  - Task time limit: 30 minutes
  - Broker: Redis

================================================================================
COMMON DEVELOPMENT WORKFLOWS
================================================================================

1. Adding a New Model Field:
   a. Edit model file: databases/models/database.py
   b. Add field: new_field = models.CharField(max_length=100, blank=True)
   c. Make migration: docker-compose exec api python manage.py makemigrations databases
   d. Apply migration: docker-compose exec api python manage.py migrate
   e. Update serializer if needed: databases/serializers/database.py
   f. Test: docker-compose exec api python manage.py test databases

2. Creating a New Model:
   a. Add to app/models.py or create app/models/new_model.py
   b. If using models/ directory, update models/__init__.py
   c. Make migration: docker-compose exec api python manage.py makemigrations app_name
   d. Apply migration: docker-compose exec api python manage.py migrate
   e. Register in admin.py if needed
   f. Create serializer, view, and URL patterns

3. Adding a New Endpoint:
   a. Create/update serializer in app/serializers.py
   b. Create view in app/views.py (ViewSet, APIView, etc.)
   c. Add URL pattern in app/urls.py
   d. Include in core/urls.py if new app
   e. Test with: docker-compose exec api python manage.py test app_name
   f. Document in Swagger (auto-generated)

4. Debugging a Feature:
   a. Check logs: docker-compose logs -f api
   b. Django shell: docker-compose exec -it api python manage.py shell
   c. Set breakpoints: import pdb; pdb.set_trace()
   d. Run specific test: docker-compose exec api python manage.py test app.tests.TestClass.test_method
   e. Check database: docker-compose exec -it postgres psql -U caissify_user -d caissify_db

5. Adding Internationalization (i18n):
   a. Mark strings for translation in code: from django.utils.translation import gettext_lazy as _
   b. Generate messages: docker exec -w /app/app_name caissify_api python ../manage.py makemessages -l es
   c. Translate strings in app_name/locale/es/LC_MESSAGES/django.po
   d. Compile messages: docker exec -w /app/app_name caissify_api python ../manage.py compilemessages
   e. Test with Accept-Language header

6. Running a Full Development Cycle:
   a. Start services: docker-compose up -d
   b. Make code changes
   c. If model changed: makemigrations → migrate
   d. Test: docker-compose exec api python manage.py test
   e. Check API: curl or Swagger UI
   f. View logs: docker-compose logs -f api
   g. Stop services: docker-compose down

================================================================================
DATABASE ACCESS
================================================================================

Using Docker:
  # PostgreSQL Shell
  docker-compose exec -it postgres psql -U caissify_user -d caissify_db
  
  # Common SQL commands:
  \dt                    # List tables
  \d+ table_name         # Describe table
  SELECT * FROM users_user LIMIT 10;
  
  # Backup
  docker-compose exec postgres pg_dump -U caissify_user caissify_db > backup.sql
  
  # Restore
  docker-compose exec -i postgres psql -U caissify_user caissify_db < backup.sql

Using pgAdmin:
  - Start with profile: docker-compose --profile tools up -d
  - Access: http://localhost:5050
  - Login: admin@caissify.local / admin
  - Connect to: postgres:5432

External Access (from host):
  - Host: localhost
  - Port: 5433 (mapped from container's 5432)
  - Database: caissify_db
  - User: caissify_user

Django Shell:
  docker-compose exec -it api python manage.py shell
  
  # Inside shell:
  from databases.models import ChessDatabase
  ChessDatabase.objects.all()
  from django.contrib.auth import get_user_model
  User = get_user_model()
  User.objects.filter(is_staff=True)

================================================================================
KEY CONVENTIONS AND PATTERNS
================================================================================

1. Model Organization:
   - Simple apps: Single models.py file
   - Complex apps: models/ directory with __init__.py importing all models
   - Always import from package level: from app.models import Model

2. URL Patterns:
   - Versioned APIs: /api/v1/, /api/v2/
   - App URLs: app/urls.py included in core/urls.py
   - Use Django REST Framework routers for ViewSets

3. Serializers:
   - Location: app/serializers.py or app/serializers/ directory
   - Naming: ModelSerializer, ModelListSerializer, ModelDetailSerializer

4. Views:
   - Prefer ViewSets over individual views for CRUD
   - Use APIView for custom logic
   - Location: app/views.py or app/views/ directory

5. Tests:
   - Unit tests: app/tests.py or app/tests/
   - Integration tests: tests/feature/
   - Use Django TestCase or DRF APITestCase
   - Use setUp() for common test data

6. Permissions:
   - Custom permissions: app/permissions.py
   - Use IsAuthenticated, IsOwner, IsOrganizer patterns

7. Pagination:
   - Custom pagination: app/pagination.py
   - Default: 25 items per page

8. Services:
   - Business logic: app/services/ directory
   - Keep views thin, put logic in services

9. Tasks:
   - Celery tasks: app/tasks.py
   - Use @shared_task decorator
   - Return meaningful results for monitoring

10. Migrations:
    - Descriptive names
    - Review before applying
    - Test rollback if changing data

================================================================================
TROUBLESHOOTING
================================================================================

Service Won't Start:
  - Check logs: docker-compose logs api
  - Check dependencies: docker-compose ps
  - Rebuild: docker-compose up --build
  - Check ports: Make sure 8000, 5433, 6380 are available

Database Connection Issues:
  - Ensure postgres service is healthy: docker-compose ps
  - Check DB_HOST=postgres (not localhost) in docker environment
  - Wait for healthcheck: depends_on with condition: service_healthy

Migration Errors:
  - Check model syntax
  - Check migration file: app/migrations/XXXX_name.py
  - Fake migration if needed: --fake
  - Rollback: migrate app_name previous_migration_number

Import Errors:
  - Check PYTHONPATH and DJANGO_SETTINGS_MODULE
  - Ensure __init__.py exists in package directories
  - Import from package level, not file level

Test Failures:
  - Check test database is created
  - Use --keepdb to reuse test database
  - Check test data setup in setUp()
  - Verify test isolation (no shared state between tests)

Celery Tasks Not Running:
  - Check worker is running: docker-compose ps worker
  - Check worker logs: docker-compose logs -f worker
  - Check Redis connection: docker-compose exec worker celery -A core inspect ping
  - Check task routes in settings.py

Performance Issues:
  - Check database indexes
  - Check N+1 queries (use select_related/prefetch_related)
  - Check Celery task queue length
  - Monitor Redis memory usage

================================================================================
USEFUL COMMANDS CHEAT SHEET
================================================================================

Docker Compose:
  docker-compose up -d                               # Start all services
  docker-compose down                                # Stop all services
  docker-compose logs -f api                         # Follow API logs
  docker-compose exec api bash                       # Shell into API container
  docker-compose restart api                         # Restart API service
  docker-compose up --build                          # Rebuild and start

Django Management:
  docker-compose exec api python manage.py migrate
  docker-compose exec api python manage.py makemigrations
  docker-compose exec api python manage.py createsuperuser
  docker-compose exec api python manage.py shell
  docker-compose exec api python manage.py test
  docker-compose exec api python manage.py collectstatic --noinput
  docker-compose exec api python manage.py showmigrations
  docker-compose exec api python manage.py dbshell

Testing:
  docker-compose exec api python manage.py test                        # All tests
  docker-compose exec api python manage.py test app_name               # App tests
  docker-compose exec api python manage.py test --keepdb               # Reuse DB
  docker-compose exec api python tests/feature/test_file.py            # Integration test

Database:
  docker-compose exec postgres psql -U caissify_user -d caissify_db   # PostgreSQL shell
  docker-compose exec postgres pg_dump -U caissify_user caissify_db > backup.sql

Celery:
  docker-compose logs -f worker                                        # Worker logs
  docker-compose exec worker celery -A core inspect active             # Active tasks
  docker-compose exec worker celery -A core status                     # Worker status
  docker-compose exec worker celery -A core inspect stats              # Task stats
  docker-compose exec worker celery -A core inspect registered         # Registered tasks

Code Quality:
  docker-compose exec api python -m pytest                             # pytest
  docker-compose exec api python -m pylint app_name                    # Linting
  docker-compose exec api python -m coverage run manage.py test       # Coverage

================================================================================
REFERENCES
================================================================================

Documentation:
  - Project docs: /docs/README.md
  - API Reference: /docs/api/
  - FIDE Integration: /docs/fide/
  - Tournament System: /docs/tournament/

Key Files:
  - Settings: core/settings.py
  - URLs: core/urls.py
  - Celery: core/celery.py
  - Docker: docker-compose.yml, Dockerfile
  - Dependencies: requirements.txt

External Resources:
  - Django: https://docs.djangoproject.com/
  - DRF: https://www.django-rest-framework.org/
  - Celery: https://docs.celeryq.dev/
  - Docker: https://docs.docker.com/

================================================================================
END OF LOCAL CONTEXT
================================================================================
