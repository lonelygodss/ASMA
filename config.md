# Simulation config & results

## 2048

### Baseline_baseline-1

* config

```python
    hierarchy = {
        HierarchyType.ACCELERATOR.value: 1,
        HierarchyType.BANK.value: 1,
        HierarchyType.TILE.value: 3,
        HierarchyType.PE.value: 16
    }
```

* result

Total subfunctions: 78
Simulation finished at time: 2432.4160000000006

### Subtile_baseline-1

* config

```python
    hierarchy = {
        HierarchyType.ACCELERATOR.value: 1,
        HierarchyType.BANK.value: 1,
        HierarchyType.TILE.value: 1,
        HierarchyType.SUBTILE.value: 16,
        HierarchyType.PE.value: 5
    }
```

* result

Total subfunctions: 78
Simulation finished at time: 877.5544615384616

### Subtile_parallel-1

* config

```python
    hierarchy = {
        HierarchyType.ACCELERATOR.value: 1,
        HierarchyType.BANK.value: 1,
        HierarchyType.TILE.value: 1,
        HierarchyType.SUBTILE.value: 13,
        HierarchyType.PE.value: 5
    }
```

* result

Total subfunctions: 65
Simulation finished at time: 547.3255384615384

## 1024

### Baseline_baseline-2

* config

```python
    hierarchy = {
        HierarchyType.ACCELERATOR.value: 1,
        HierarchyType.BANK.value: 1,
        HierarchyType.TILE.value: 9,
        HierarchyType.PE.value: 16
    }
```

* result

Total subfunctions: 201
Simulation finished at time: 4297.584000000003

### Subtile_baseline-2

* config

```python
    hierarchy = {
        HierarchyType.ACCELERATOR.value: 1,
        HierarchyType.BANK.value: 1,
        HierarchyType.TILE.value: 1,
        HierarchyType.SUBTILE.value: 44,
        HierarchyType.PE.value: 5
    }
```

* result

Total subfunctions: 201
Simulation finished at time: 2173.144000000001

### Subtile_parallel-2

* config

```python
    hierarchy = {
        HierarchyType.ACCELERATOR.value: 1,
        HierarchyType.BANK.value: 1,
        HierarchyType.TILE.value: 1,
        HierarchyType.SUBTILE.value: 44,
        HierarchyType.PE.value: 5
    }
```

* result

Total subfunctions: 183
Simulation finished at time: 1053.4465454545452

## 512

### Baseline_baseline-3

* config

```python
    hierarchy = {
        HierarchyType.ACCELERATOR.value: 1,
        HierarchyType.BANK.value: 1,
        HierarchyType.TILE.value: 9,
        HierarchyType.PE.value: 16
    }
```

* result

Total subfunctions: 201
Simulation finished at time: 4297.584000000003

### Subtile_baseline-3

* config

```python
    hierarchy = {
        HierarchyType.ACCELERATOR.value: 1,
        HierarchyType.BANK.value: 1,
        HierarchyType.TILE.value: 1,
        HierarchyType.SUBTILE.value: 44,
        HierarchyType.PE.value: 5
    }
```

* result

Total subfunctions: 201
Simulation finished at time: 2173.144000000001

### Subtile_parallel-3

* config

```python
    hierarchy = {
        HierarchyType.ACCELERATOR.value: 1,
        HierarchyType.BANK.value: 1,
        HierarchyType.TILE.value: 3,
        HierarchyType.SUBTILE.value: 64,
        HierarchyType.PE.value: 5
    }
```

* result

Total subfunctions: 627
Simulation finished at time: 13456.259999999835
