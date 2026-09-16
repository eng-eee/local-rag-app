# PostgreSQL MVCC

PostgreSQL uses Multi-Version Concurrency Control (MVCC) to provide
transaction isolation.

When a row is updated, PostgreSQL does not simply overwrite the existing
row version. Instead, a new row version is created.

Different transactions can therefore see different row versions depending
on their transaction visibility.

VACUUM is responsible for removing row versions that are no longer visible
to any active transaction.
