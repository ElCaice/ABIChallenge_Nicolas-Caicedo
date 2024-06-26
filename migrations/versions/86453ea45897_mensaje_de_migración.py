"""Mensaje de migración

Revision ID: 86453ea45897
Revises: c269606bc130
Create Date: 2024-04-10 04:18:04.296826

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '86453ea45897'
down_revision = 'c269606bc130'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('prediccion', schema=None) as batch_op:
        batch_op.drop_constraint('prediccion_prediction_key', type_='unique')

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('prediccion', schema=None) as batch_op:
        batch_op.create_unique_constraint('prediccion_prediction_key', ['prediction'])

    # ### end Alembic commands ###
