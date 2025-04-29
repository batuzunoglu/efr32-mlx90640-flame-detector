//

#ifndef MLX90640_I2C_H
#define MLX90640_I2C_H
#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
#include "sl_status.h"
#include "sl_i2cspm.h"

/***************************************************************************//**
 * @brief
 * Assigns an I2CSPM instance for the driver to use
 *
 * @param[in] i2cspm_instace - Pointer to the I2CSPM instance
 ******************************************************************************/
sl_status_t mlx90640_I2C_Init(sl_i2cspm_t *i2cspm_instance);

/***************************************************************************//**
 * @brief
 * Issues an I2C general reset
 ******************************************************************************/
sl_status_t mlx90640_I2CGeneralReset(void);

/***************************************************************************//**
 * @brief
 * Initiates an I2C read of the device
 *
 * @param[in] startAddress - EEPROM memory address of the device to read out from
 * @param[in] nMemAddressRead - Length of the requested data
 * @param[out] data - pointer to an array where the received data will be stored
 ******************************************************************************/
sl_status_t mlx90640_I2CRead(uint16_t startAddress, uint16_t nMemAddressRead, uint16_t *data);

/***************************************************************************//**
 * @brief
 * Initiates an I2C write to the device
 *
 * @param[in] writeAddress - EEPROM memory address of the device to write to
 * @param[in] data - 16bit data to send to the device
 ******************************************************************************/
sl_status_t mlx90640_I2CWrite(uint16_t writeAddress, uint16_t data);

/***************************************************************************//**
 * @brief
 * Sets I2C base frequency to a given setting
 *
 * @param[in] freq - new frequency in Hz to set for the I2C base frequency
 ******************************************************************************/
void mlx90640_I2CFreqSet(int freq);
#ifdef __cplusplus
}
#endif
#endif // MLX90640_I2C_H
